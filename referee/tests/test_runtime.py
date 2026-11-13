"""The per-GPU filenames of G1 (one commit on three Colab GPUs) and the runtime record."""
from referee import runtime


def test_gpu_slugs():
    assert runtime.gpu_slug("NVIDIA A100-SXM4-40GB") == "a100"
    assert runtime.gpu_slug("NVIDIA A100-SXM4-80GB") == "a100"
    assert runtime.gpu_slug("NVIDIA H100 80GB HBM3") == "h100"
    assert runtime.gpu_slug("NVIDIA RTX PRO 6000 Blackwell Server Edition") == "rtxpro6000"
    assert runtime.gpu_slug("Tesla T4") == "t4"
    assert runtime.gpu_slug("NVIDIA L4") == "l4"
    assert runtime.gpu_slug("") == "nogpu"


def test_environment_and_save(tmp_path, monkeypatch):
    env = runtime.environment()
    assert {"gpu", "gpu_slug", "driver", "cuda_driver_api", "torch_cuda", "rapids", "packages"} <= set(env)
    monkeypatch.setattr(runtime, "gpu_slug", lambda name=None: "h100")
    monkeypatch.setattr(runtime, "commit", lambda: "abc1234")
    src = tmp_path / "results.json"
    assert runtime.save(str(src), folder=str(tmp_path)) is None
    src.write_text("{}")
    assert runtime.save(str(src), "s9", folder=str(tmp_path)) == str(tmp_path / "s3-search-results-s9-abc1234-h100.json")
