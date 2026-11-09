"""The freshness harness on the reference index: it passes a correct updater and fails a lossy one."""
from referee import freshness, loader, reference


def test_patches_are_visible_and_equal_a_rebuild():
    ds = loader.load_dataset("toy")
    idx = reference.build_index(ds.catalogue, ds.mapping)
    rep = freshness.check_freshness(idx, ds.catalogue, ds.mapping, freshness.patch_hook(idx), reference.build_index,
                                    n_changes=10_000)
    assert rep["passed"], rep
    assert rep["visible"] > 0 and rep["visible_ms"]["p99"] >= 0


def test_a_lossy_updater_fails():
    ds = loader.load_dataset("toy")
    idx = reference.build_index(ds.catalogue, ds.mapping)
    good = freshness.patch_hook(idx)
    lossy = lambda changes: good(changes[:-1])  # noqa: E731  (drops the last change of every batch)
    rep = freshness.check_freshness(idx, ds.catalogue, ds.mapping, lossy, reference.build_index,
                                    n_changes=300, timeout_s=0.05)
    assert not rep["passed"] and (rep["failures"] or not rep["equal_to_rebuild"])
