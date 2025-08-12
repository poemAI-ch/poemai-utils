import os

from poemai_utils.dir_hash import compute_directory_hash


def test_hash_stable(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.txt").write_bytes(b"hello")
    (d / "sub").mkdir()
    (d / "sub" / "b.bin").write_bytes(b"\x00\x01")

    h1 = compute_directory_hash(d)
    h2 = compute_directory_hash(d)
    assert h1 == h2


def test_name_change_changes_hash(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a").write_text("x")
    h1 = compute_directory_hash(d)
    (d / "b").write_text("x")
    (d / "a").unlink()
    h2 = compute_directory_hash(d)
    assert h1 != h2


def test_symlink_target_affects_hash(tmp_path):
    d = tmp_path
    (d / "file").write_text("x")
    os.symlink("file", d / "ln")
    h1 = compute_directory_hash(d)
    os.unlink(d / "ln")
    os.symlink("file2", d / "ln")
    h2 = compute_directory_hash(d)
    assert h1 != h2
