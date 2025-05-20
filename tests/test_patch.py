from pathlib import Path

from elastic_trader.evolve.alpha_loop import apply_patch


def test_apply_patch(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("foo\n")
    patch = """--- a/a.txt\n+++ b/a.txt\n@@\n-foo\n+bar\n"""
    apply_patch(patch, tmp_path)
    assert target.read_text() == "bar\n"

