import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

from torch_geometric.nn import SGConv

class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)

class GEARS_Model(torch.nn.Module):
    """
    GEARS
    """

    def __init__(self, args):
        super(GEARS_Model, self).__init__()
        self.args = args       
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.hidden_size = hidden_size
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.cell_fitness_pred = args['cell_fitness_pred']
        self.pert_emb_lambda = 0.2
        
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)
           
        # gene/globel perturbation embedding dictionary lookup            
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)
        
        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        
        # gene co-expression GNN
        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))
        
        ### perturbation gene ontology GNN
        self.G_sim = args['G_go'].to(args['device'])
        self.G_sim_weight = args['G_go_weight'].to(args['device'])

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))
        
        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')
        
        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)
        
        # Cross gene MLP
        self.cross_gene_state = MLP([self.num_genes, hidden_size,
                                     hidden_size])
        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                           hidden_size+1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)
        
        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        
        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        
        #if self.cell_fitness_pred:
        self.cell_fitness_mlp = MLP([self.num_genes, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        
        if args['model_type'] == 'maeautobin':
            from modules.encoders import MAEAutobinencoder
            self.singlecell_model = MAEAutobinencoder(args, hidden_size=hidden_size)
            self.pretrained = True
            print('Single cell model load success! model type: MAEAutobin')
        elif args['model_type'] == 'API':
            def API(x):
                return torch.rand(x.shape[0], x.shape[1]-1,hidden_size).to(x.device)
            self.singlecell_model = API
            self.pretrained = True
        else:
            self.pretrained = False
            print('No Single cell model load!')

    def forward(self, data):
        x, pert_idx = data.x, data.pert_idx
        if self.no_perturb:
            out = x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique())

            ## get base gene embeddings
            if self.pretrained:
                pre_in = x.clone().reshape(num_graphs, self.num_genes+1)
                x = x.reshape(num_graphs, self.num_genes+1)[:,:-1]
                # omni-scfm memory patch (faithful, NO numerical change). When the
                # scFoundation encoder is frozen (finetune_method='frozen' sets every
                # singlecell_model param requires_grad=False; see gears.py:351-352), this
                # forward STILL retains the entire encoder activation graph, because
                # modules/mae_autobin.py:126 calls x.requires_grad_() (for an attention-map
                # feature we never use in training). For a ~19k-gene transformer at batch 6
                # that's ~15 GB of a gradient graph that can never be used — it is exactly
                # what OOMs a 24 GB GPU at the paper's batch_size=6.
                #
                # Running the FROZEN encoder under no_grad is provably identical:
                #   * no_grad never changes forward *values* -> emb is bit-identical
                #     (verified: m(x) == no_grad m(x));
                #   * the encoder params are frozen, so they get no gradient either way;
                #   * the trainable GEARS head's gradients depend only on emb's VALUE, not
                #     on whether emb tracks grad -> every trainable grad is unchanged.
                # It only drops the unused encoder graph. Net: scFoundation trains MORE
                # FRUGALLY than the original paper here (the paper relied on bigger GPUs) —
                # batch=6 fits a 24 GB L4 with no change to results. The finetune path
                # (which *does* backprop the encoder) is deliberately left untouched.
                if self.args.get('finetune_method') == 'frozen':
                    with torch.no_grad():
                        emb = self.singlecell_model(pre_in)
                else:
                    emb = self.singlecell_model(pre_in)
                if 'v1' in self.args['mode']:
                    pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
                elif 'v2' in self.args['mode']:
                    pos_emb = emb.view(-1, self.hidden_size)
                else:
                    print('error!')
                    exit()
                emb = emb.view(-1, self.hidden_size)
            else:
                x = x.reshape(num_graphs, self.num_genes+1)[:,:-1]
                emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
                pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)        
            # pos_emb = emb
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)

            ## get perturbation index and embeddings

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T

            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))        

            ## augment global perturbation embedding with GNN
            for idx, layer in enumerate(self.sim_layers):
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()

            ## add global perturbation embedding to each gene in each cell in the batch
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)

            ## apply the first MLP
            base_emb = self.transform(base_emb)        
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis = 2)
            out = w + self.indv_b1

            # Cross gene
            cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)

            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2        
            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)

            ## uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)
            
            if self.cell_fitness_pred:
                return torch.stack(out), self.cell_fitness_mlp(torch.stack(out))
            
            return torch.stack(out)
        
