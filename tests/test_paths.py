from pathlib import Path

from t3_telemetry import paths
from t3_telemetry.cli import build_parser, main, t3_state_problem
from t3_telemetry.collector import default_roots
from t3_telemetry.db_agy import find_agy_db_path


def test_defaults_follow_home(isolated_home: Path):
    assert paths.store_path() == isolated_home / ".t3-telemetry/telemetry.sqlite"
    assert paths.t3_dir() == isolated_home / ".t3/userdata"
    assert paths.antigravity_cli_dir() == isolated_home / ".gemini/antigravity-cli"


def test_env_overrides(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("T3_TELEMETRY_STORE", str(tmp_path / "s.sqlite"))
    monkeypatch.setenv("T3CODE_HOME", str(tmp_path / "t3home"))
    monkeypatch.setenv("ANTIGRAVITY_CLI_DIR", str(tmp_path / "agy"))
    assert paths.store_path() == tmp_path / "s.sqlite"
    assert paths.t3_dir() == tmp_path / "t3home/userdata"
    assert default_roots(tmp_path / "t3").antigravity_dirs[-1] == tmp_path / "agy/conversations"
    monkeypatch.setenv("T3_TELEMETRY_T3_DIR", str(tmp_path / "explicit"))
    assert paths.t3_dir() == tmp_path / "explicit"


def test_flags_beat_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("T3_TELEMETRY_STORE", str(tmp_path / "env.sqlite"))
    monkeypatch.setenv("T3_TELEMETRY_T3_DIR", str(tmp_path / "env-t3"))
    args = build_parser().parse_args(["--store", str(tmp_path / "flag.sqlite"), "--t3-dir", str(tmp_path / "flag-t3")])
    assert paths.store_path(args.store) == tmp_path / "flag.sqlite"
    assert paths.t3_dir(args.t3_dir) == tmp_path / "flag-t3"


def test_antigravity_cli_dir_used_for_lookup(tmp_path: Path, monkeypatch):
    agy = tmp_path / "agy"
    (agy / "conversations").mkdir(parents=True)
    (agy / "conversations/abc.db").touch()
    monkeypatch.setenv("ANTIGRAVITY_CLI_DIR", str(agy))
    assert find_agy_db_path("abc", tmp_path / "no-t3") == agy / "conversations/abc.db"


def test_missing_t3_is_explained(tmp_path: Path, capsys):
    assert "--t3-dir" in t3_state_problem(tmp_path / "nope")
    code = main(["--sessions", "--t3-dir", str(tmp_path / "nope")])
    assert code == 0
    err = capsys.readouterr().err
    assert "T3 state not found" in err and str(tmp_path / "nope") in err


def test_sync_without_t3_still_reads_native_sessions(tmp_path: Path, monkeypatch, capsys):
    project = tmp_path / "claude/projects/p"
    project.mkdir(parents=True)
    (project / "s1.jsonl").write_text(
        '{"type":"assistant","timestamp":"2026-11-03T12:00:01Z","sessionId":"s1",'
        '"message":{"id":"m1","model":"claude-opus-5-5","usage":{"input_tokens":10,"output_tokens":5}}}\n'
    )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    assert main(["--sync", "--t3-dir", str(tmp_path / "nope")]) == 0
    assert "steps_written=1" in capsys.readouterr().out
