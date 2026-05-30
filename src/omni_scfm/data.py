"""Dataset preprocessing helpers: GEARS-faithful train/val/test splitting."""

from __future__ import annotations

from ._vendor.gears_split import DataSplitter


def simulation_split(adata, seed: int) -> dict[str, list[str]]:
    """Reproduce GEARS ``PertData.prepare_split(split='simulation', seed=seed)``.

    Uses the vendored :class:`DataSplitter` with the same defaults GEARS'
    ``prepare_split`` applies for the ``'simulation'`` split type
    (``train_gene_set_size=0.75``, ``combo_seen2_train_frac=0.75``), then builds
    the ``set2conditions`` mapping exactly as GEARS does.

    Args:
        adata: AnnData whose ``obs`` has a ``condition`` column (incl. ``ctrl``).
        seed: split seed (the paper uses 1,2 for single- and 1..5 for
            double-perturbation datasets).

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` of condition strings.
    """
    ds = DataSplitter(adata, split_type="simulation")
    adata, _subgroup = ds.split_data(
        train_gene_set_size=0.75,
        combo_seen2_train_frac=0.75,
        seed=seed,
        test_perts=None,
        only_test_set_perts=False,
    )

    # Identical to GEARS PertData.prepare_split.
    set2conditions = dict(
        adata.obs.groupby("split").agg({"condition": lambda x: x}).condition
    )
    set2conditions = {k: v.unique().tolist() for k, v in set2conditions.items()}
    return set2conditions
