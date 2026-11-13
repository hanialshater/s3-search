"""Markers for the referee's own tests: `slow` tests run with the judges, not in `referee fast`."""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: the judges' full self-tests; deselected by `referee fast`")
