import pytest

_PATH_VARS = ("T3_TELEMETRY_STORE", "T3_TELEMETRY_T3_DIR", "T3CODE_HOME",
              "CLAUDE_CONFIG_DIR", "CODEX_HOME", "ANTIGRAVITY_CLI_DIR")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    """Keep tests away from the real ~/.claude, ~/.codex, ~/.t3 and telemetry store."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    for name in _PATH_VARS:
        monkeypatch.delenv(name, raising=False)
    return home
