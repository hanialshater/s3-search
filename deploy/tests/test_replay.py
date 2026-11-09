"""The e2e replay tool, against serving with the fake engine (so it is exercised before kind ever runs)."""
import threading

from deploy.e2e.replay import main, replay
from serving.app import create_app, make_server
from serving.config import ServingConfig
from serving.tests.fakes import MAPPING, fake_ports, write_source


def test_replay_counts_failures_and_exit_status(tmp_path, capsys):
    app = create_app(ServingConfig(data_dir=str(tmp_path / "d")), fake_ports())
    src = write_source(tmp_path / "s.jsonl")
    assert app.handle("PUT", "/indexes/toy", __import__("json").dumps(
        {"mapping": MAPPING, "source": {"path": src}}).encode()).status == 200
    server = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        out = replay(url, "toy", qps=100, duration=0.5, field="title")
        assert out["sent"] == 50 and out["failed"] == 0 and out["p99_ms"] is not None
        assert main(["--url", url, "--index", "toy", "--qps", "50", "--duration", "0.2", "--field", "title"]) == 0
        assert main(["--url", url, "--index", "missing", "--qps", "50", "--duration", "0.2"]) == 1   # 404s are failures
    finally:
        server.shutdown()
        server.server_close()
        app.close()
