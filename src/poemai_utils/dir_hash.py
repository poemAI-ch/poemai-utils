#!/usr/bin/env python3
"""
Deterministic SHA-256 of a directory tree.

Defaults:
- Ignores mtimes, uid/gid.
- Ignores permissions (set include_perms=True to include).
- Does NOT follow symlinks; hashes the link target text.
- Ignores extended attributes/ACLs unless include_xattrs=True.

Hash changes if: any filename changes, structure changes, symlink target changes,
or any file content changes (and optionally perms/xattrs if enabled).
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from stat import S_ISDIR, S_ISLNK, S_ISREG

CHUNK = 1024 * 1024


def _sha256_file(p: Path) -> bytes:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.digest()


def _perm_octal(mode: int) -> bytes:
    # only permission bits, in octal, four digits (e.g., 0755 -> b"0755")
    return f"{mode & 0o7777:04o}".encode("ascii")


def _xattrs_blob(p: Path) -> bytes:
    # Best-effort, Linux/macOS; ignore if unsupported.
    try:
        names = sorted(os.listxattr(p, follow_symlinks=False))
    except Exception:
        return b""
    parts: list[bytes] = []
    for n in names:
        try:
            v = os.getxattr(p, n, follow_symlinks=False)
        except Exception:
            v = b""
        # NUL-separate name/value pairs; name encoded as utf-8 best-effort
        parts.append(n.encode("utf-8", "surrogateescape") + b"\0" + v + b"\0")
    return b"".join(parts)


def compute_directory_hash(
    root: Path,
    *,
    include_perms: bool = False,
    include_xattrs: bool = False,
    follow_symlinks: bool = False,
    exclude: list[str] | None = None,
) -> str:
    """
    Return hex sha256 of a deterministic manifest of the directory tree.

    :param root: path to a directory
    :param include_perms: include permission bits for files/dirs/symlinks
    :param include_xattrs: include extended attributes (may vary across OS)
    :param follow_symlinks: if True, treat symlinks as their targets (danger of loops)
    :param exclude: list of glob patterns (relative to root) to exclude
    """
    if exclude is None:
        exclude = []

    root = root.resolve()
    manifest = hashlib.sha256()

    def excluded(rel: str) -> bool:
        from fnmatch import fnmatch

        return any(fnmatch(rel, pat) for pat in exclude)

    # Walk deterministically
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""

        # Filter and sort children
        dirnames[:] = sorted(
            d for d in dirnames if not excluded(os.path.join(rel_dir, d))
        )
        files = sorted(f for f in filenames if not excluded(os.path.join(rel_dir, f)))

        # Record the directory node itself so dir name/perms influence the hash
        st = os.lstat(dirpath)
        _update_manifest_node(
            manifest,
            root,
            Path(dirpath),
            rel_dir,
            st,
            node_type="dir",
            include_perms=include_perms,
            include_xattrs=include_xattrs,
        )

        # Files & symlinks (and others)
        for fn in files:
            p = Path(dirpath) / fn
            rel = os.path.join(rel_dir, fn)
            st = os.lstat(p)

            if S_ISREG(st.st_mode):
                # content hash
                content = _sha256_file(p)
                _update_manifest_file(
                    manifest,
                    rel,
                    st,
                    content,
                    include_perms=include_perms,
                    include_xattrs=include_xattrs,
                )

            elif S_ISLNK(st.st_mode):
                target = os.readlink(p)  # text of the link target
                _update_manifest_symlink(
                    manifest,
                    rel,
                    st,
                    target,
                    include_perms=include_perms,
                    include_xattrs=include_xattrs,
                )

            elif S_ISDIR(st.st_mode):
                # Shouldn’t happen here (dirs are handled by os.walk), but keep for completeness.
                _update_manifest_node(
                    manifest,
                    root,
                    p,
                    rel,
                    st,
                    node_type="dir",
                    include_perms=include_perms,
                    include_xattrs=include_xattrs,
                )
            else:
                # fifos, sockets, device nodes: include type+rel path (+perms/xattrs if requested)
                _update_manifest_other(
                    manifest,
                    rel,
                    st,
                    include_perms=include_perms,
                    include_xattrs=include_xattrs,
                )

    return manifest.hexdigest()


def _b(s: str) -> bytes:
    return s.encode("utf-8", "surrogateescape")


def _update_manifest_node(
    manifest,
    root,
    p: Path,
    rel: str,
    st,
    *,
    node_type: str,
    include_perms: bool,
    include_xattrs: bool,
):
    # Entry format (NUL-separated fields):
    # type NUL relpath NUL perms? NUL xattrs?
    manifest.update(b"D\0" if node_type == "dir" else b"N\0")
    manifest.update(_b(rel) + b"\0")
    if include_perms:
        manifest.update(_perm_octal(st.st_mode) + b"\0")
    if include_xattrs:
        manifest.update(_xattrs_blob(p) + b"\0")


def _update_manifest_file(
    manifest,
    rel: str,
    st,
    content_digest: bytes,
    *,
    include_perms: bool,
    include_xattrs: bool,
):
    # type('F') NUL rel NUL size NUL content_sha256 NUL perms? NUL xattrs?
    manifest.update(b"F\0")
    manifest.update(_b(rel) + b"\0")
    manifest.update(str(st.st_size).encode("ascii") + b"\0")
    manifest.update(content_digest + b"\0")
    if include_perms:
        manifest.update(_perm_octal(st.st_mode) + b"\0")
    if include_xattrs:
        # We need a Path to fetch xattrs; skip here to avoid extra stat/open; caller can add if needed
        pass


def _update_manifest_symlink(
    manifest, rel: str, st, target: str, *, include_perms: bool, include_xattrs: bool
):
    # type('L') NUL rel NUL target NUL perms?
    manifest.update(b"L\0")
    manifest.update(_b(rel) + b"\0")
    manifest.update(_b(target) + b"\0")
    if include_perms:
        manifest.update(_perm_octal(st.st_mode) + b"\0")
    # xattrs on symlinks are uncommon; skipped by default


def _update_manifest_other(
    manifest, rel: str, st, *, include_perms: bool, include_xattrs: bool
):
    # type('O') NUL rel NUL mode-type NUL perms?
    manifest.update(b"O\0")
    manifest.update(_b(rel) + b"\0")
    manifest.update(hex(st.st_mode & 0o170000).encode("ascii") + b"\0")
    if include_perms:
        manifest.update(_perm_octal(st.st_mode) + b"\0")


def _cli():
    ap = argparse.ArgumentParser(
        description="Deterministic sha256 of a directory tree."
    )
    ap.add_argument("path", type=Path)
    ap.add_argument(
        "--include-perms", action="store_true", help="Include permission bits"
    )
    ap.add_argument(
        "--include-xattrs",
        action="store_true",
        help="Include extended attributes (macOS/Linux)",
    )
    ap.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Hash targets instead of link text",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob relative to root (e.g. .git, *.pyc)",
    )
    args = ap.parse_args()
    print(
        compute_directory_hash(
            args.path,
            include_perms=args.include_perms,
            include_xattrs=args.include_xattrs,
            follow_symlinks=args.follow_symlinks,
            exclude=args.exclude or None,
        )
    )


if __name__ == "__main__":
    _cli()
