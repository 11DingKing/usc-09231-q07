"""Tests for the unified archive member iteration and temporary
extraction-directory lifecycle in `beets.importer.archives`.

These tests deliberately import only the dependency-free `archives`
module (standard library plus optional `py7zr`), so the archive logic
can be verified without the full beets runtime.
"""

from __future__ import annotations

import io
import os
import tarfile
import time
import zipfile
from pathlib import Path

import pytest

from beets.importer import archives

# A fixed set of structures exercised against every format:
#   flat       - members at the archive root
#   nested     - files several directories deep plus an empty directory
#   wrapper    - everything beneath a single top-level folder
#   emptydir   - an archive that only contains an empty directory
STRUCTURES: dict[str, dict[str, bytes]] = {
    "flat": {
        "a.mp3": b"aaa",
        "b.mp3": b"bbb",
        "c.log": b"log",
    },
    "nested": {
        "a.mp3": b"aaa",
        "Artist/cover.jpg": b"jpg",
        "Artist/Album/Disc 1/t1.mp3": b"111",
        "Artist/Album/Disc 2/t2.mp3": b"222",
    },
    "wrapper": {
        "Release/t1.mp3": b"111",
        "Release/sub/t2.mp3": b"222",
    },
    "emptydir": {},
}
NESTED_EMPTY_DIR = "Artist/empty dir"

pytestmark = pytest.mark.skipif(
    not archives._module_available("py7zr"),
    reason="py7zr not installed",
)


def _materialize(root: Path, files: dict[str, bytes], empty_dirs=()):
    for name, data in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    for d in empty_dirs:
        (root / d).mkdir(parents=True, exist_ok=True)


def _expected_dirs(files: dict[str, bytes], empty_dirs=()) -> set[str]:
    dirs = set()
    for name in files:
        parts = name.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    dirs.update(empty_dirs)
    return dirs


def _make_zip(path: Path, files: dict[str, bytes], empty_dirs=()):
    with zipfile.ZipFile(path, "w") as z:
        # Explicit directory entries, including the empty one.
        for d in sorted(_expected_dirs(files, empty_dirs)):
            z.writestr(zipfile.ZipInfo(d + "/"), b"")
        for i, (name, data) in enumerate(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2000, 1, 1, 0, 0, i))
            z.writestr(info, data)


def _make_tar(path: Path, files: dict[str, bytes], empty_dirs=()):
    old = time.mktime((2000, 1, 1, 0, 0, 0, 0, 0, -1))
    with tarfile.open(path, "w") as t:
        for d in sorted(_expected_dirs(files, empty_dirs)):
            info = tarfile.TarInfo(d)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            info.mtime = old
            t.addfile(info)
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = old
            t.addfile(info, io.BytesIO(data))


def _make_7z(path: Path, files: dict[str, bytes], empty_dirs=()):
    import py7zr

    root = path.parent / f"{path.stem}_src"
    _materialize(root, files, empty_dirs)
    with py7zr.SevenZipFile(path, "w") as z:
        for name in files:
            z.write(root / name, name)
        # py7zr omits implicit parent directories, so add every directory
        # explicitly; empty ones would otherwise be lost entirely.
        for d in sorted(_expected_dirs(files, empty_dirs)):
            z.write(root / d, d)


MAKERS = {
    "zip": _make_zip,
    "tar": _make_tar,
    "7z": _make_7z,
}
SUFFIX = {"zip": ".zip", "tar": ".tar", "7z": ".7z"}


@pytest.fixture
def work(tmp_path):
    return tmp_path


def _build(fmt, structure, tmp_path, *, name=None):
    files = STRUCTURES[structure]
    empty_dirs = (NESTED_EMPTY_DIR,) if structure == "nested" else ()
    p = tmp_path / (name or f"archive{SUFFIX[fmt]}")
    MAKERS[fmt](p, files, empty_dirs)
    return p, files, empty_dirs


@pytest.mark.parametrize("fmt", ["zip", "tar", "7z"])
@pytest.mark.parametrize("structure", ["flat", "nested", "wrapper", "emptydir"])
def test_members_cover_every_file_and_directory(fmt, structure, tmp_path):
    """Iteration must not miss members, including empty directories."""
    path, files, empty_dirs = _build(fmt, structure, tmp_path)
    with archives.open_archive(str(path)) as ar:
        members = list(ar.members())

    member_names = {m.name.rstrip("/") for m in members}
    for f in files:
        assert f in member_names, f"{fmt}/{structure} missed file {f}"
    for d in _expected_dirs(files, empty_dirs):
        assert d in member_names, f"{fmt}/{structure} missed dir {d}"

    expected_all = set(files) | _expected_dirs(files, empty_dirs)
    for m in members:
        assert m.name.rstrip("/") in expected_all
        assert isinstance(m.is_dir, bool)


def test_member_iteration_preserves_archive_order(work):
    """The unified iterator yields the format's on-disk member order."""
    ordered = ["z.mp3", "a.mp3", "m.mp3"]
    p = work / "ordered.zip"
    with zipfile.ZipFile(p, "w") as z:
        for i, name in enumerate(ordered):
            z.writestr(
                zipfile.ZipInfo(name, date_time=(2000, 1, 1, 0, 0, i)), b"x"
            )
    with archives.open_archive(str(p)) as ar:
        assert [m.name for m in ar.members()] == ordered


@pytest.mark.parametrize("fmt", ["zip", "tar", "7z"])
def test_member_order_is_archive_order_for_every_format(fmt, work):
    """Members come back in insertion order, so the import pipeline sees
    the same ordering regardless of format (no per-format reordering)."""
    ordered = ["z_first.mp3", "nested/a_middle.mp3", "a_last.mp3"]
    p = work / f"ordered{SUFFIX[fmt]}"
    MAKERS[fmt](p, {n: b"x" for n in ordered})
    with archives.open_archive(str(p)) as ar:
        names = [m.name for m in ar.members() if not m.is_dir]
    assert names == ordered, (fmt, names)


@pytest.mark.parametrize("fmt", ["zip", "tar", "7z"])
@pytest.mark.parametrize("structure", ["flat", "nested", "wrapper"])
def test_extraction_layout_and_mtimes(fmt, structure, tmp_path):
    path, files, empty_dirs = _build(fmt, structure, tmp_path)
    with archives.open_archive(str(path)) as ar:
        expected_mtime = {
            m.name.rstrip("/"): m.mtime
            for m in ar.members()
            if not m.is_dir and m.mtime is not None
        }

    with archives.ExtractedArchive(str(path), dir=tmp_path) as ex:
        root = Path(ex.path)
        assert root.parent == tmp_path
        for name, data in files.items():
            f = root / name
            assert f.is_file(), f"{fmt}/{structure}: missing {name}"
            assert f.read_bytes() == data
        for d in [*_expected_dirs(files, empty_dirs), *empty_dirs]:
            assert (root / d).is_dir(), f"{fmt}/{structure}: missing dir {d}"
        for name, mtime in expected_mtime.items():
            assert abs((root / name).stat().st_mtime - mtime) < 2, (
                f"{fmt}: mtime not restored for {name}"
            )
    # Context exit always removes the directory.
    assert not root.exists()


def test_nested_empty_directory_is_extracted(work):
    path, _, empty_dirs = _build("7z", "nested", work)
    with archives.ExtractedArchive(str(path)) as ex:
        assert (Path(ex.path) / NESTED_EMPTY_DIR).is_dir()


def test_empty_archive_extracts_and_cleans_up(work):
    for fmt, make in (
        ("zip", lambda p: zipfile.ZipFile(p, "w").close()),
        ("tar", lambda p: tarfile.open(p, "w").close()),
    ):
        p = work / f"empty{SUFFIX[fmt]}"
        make(p)
        ex = archives.ExtractedArchive(str(p), dir=work)
        ex.create()
        tmp = Path(ex.path)
        assert tmp.is_dir()
        assert not ex.has_remaining_files()
        ex.cleanup()
        assert not tmp.exists()
        ex.cleanup()  # idempotent


def test_cleanup_removes_tempdir_in_all_modes(work):
    """Mirror of TestRmTemp.test_tempdir_removed_in_all_modes at the
    lifecycle layer: cleanup never depends on copy/move flags."""
    p = work / "a.zip"
    _make_zip(p, {"f.mp3": b"x"})
    for _ in range(3):
        ex = archives.ExtractedArchive(str(p), dir=work)
        ex.create()
        tmp = Path(ex.path)
        assert tmp.is_dir()
        ex.cleanup()
        assert not tmp.exists()


def test_corrupt_archive_rolls_back_tempdir(work):
    p = work / "bad.zip"
    p.write_bytes(b"this is not a zip")
    # A reader exists for zip, so magic-byte failure means "not an
    # archive" rather than an extension guess.
    assert archives.is_archive(str(p)) is False

    created = set(os.listdir(work))
    with pytest.raises(archives.ArchiveError):
        archives.ExtractedArchive(str(p), dir=work).create()
    # No beets-* temporary directory survives the failure.
    assert set(os.listdir(work)) == created


def test_mid_extraction_failure_rolls_back(work, monkeypatch):
    p = work / "a.zip"
    _make_zip(p, {"f.mp3": b"x"})

    class BoomArchive(archives.ZipArchive):
        def extractall(self, path):
            super().extractall(path)
            raise RuntimeError("boom midway")

    monkeypatch.setattr(
        archives,
        "ARCHIVE_TYPES",
        (
            archives.ArchiveType(
                "zip",
                (".zip",),
                None,
                archives.is_zipfile,
                BoomArchive,
            ),
        ),
    )

    created = set(os.listdir(work))
    with pytest.raises(RuntimeError, match="boom midway"):
        archives.ExtractedArchive(str(p), dir=work).create()
    assert set(os.listdir(work)) == created


def test_missing_dependency_clear_error_and_rollback(work, monkeypatch):
    """A recognised 7z whose reader is unavailable must raise a clear
    ArchiveDependencyError and not leave a temp directory behind."""
    p = work / "a.7z"
    _make_7z(p, {"f.mp3": b"x"})

    # Simulate py7zr never being installed. ``get_archive_type`` checks
    # availability before attempting to open the reader, so patching the
    # availability probe is enough (and avoids import-machinery hacks).
    real_available = archives._module_available
    monkeypatch.setattr(
        archives,
        "_module_available",
        lambda name: False if name == "py7zr" else real_available(name),
    )

    # Extension fallback still recognises the file.
    assert archives.is_archive(str(p))

    with pytest.raises(archives.ArchiveDependencyError, match="py7zr"):
        with archives.open_archive(str(p)):
            pass

    created = set(os.listdir(work))
    with pytest.raises(archives.ArchiveDependencyError):
        archives.ExtractedArchive(str(p), dir=work).create()
    assert set(os.listdir(work)) == created


def test_rar_without_package_recognised_but_not_openable(work):
    """rarfile is genuinely optional: without it, a .rar file is still
    detected (by extension) but opening names the missing package."""
    if archives._module_available("rarfile"):
        pytest.skip("rarfile installed")
    p = work / "x.rar"
    p.write_bytes(b"Rar!\x1a\x07\x00garbage")
    assert archives.is_archive(str(p))
    with pytest.raises(archives.ArchiveDependencyError, match="rarfile"):
        with archives.open_archive(str(p)):
            pass


def test_magic_detection_ignores_extension(work):
    p = work / "renamed.dat"
    _make_zip(p, {"f.mp3": b"x"})
    assert archives.is_archive(str(p))


def test_unknown_file_is_not_archive(work):
    p = work / "notes.txt"
    p.write_bytes(b"hello")
    assert not archives.is_archive(str(p))
    with pytest.raises(archives.ArchiveError):
        with archives.open_archive(str(p)):
            pass


def test_has_remaining_files_reflects_moved_away_members(work):
    """The move-mode 'all imported?' decision: False once every extracted
    file has gone, even if empty directories remain."""
    p = work / "a.zip"
    _make_zip(p, {"f.mp3": b"x"})
    ex = archives.ExtractedArchive(str(p), dir=work)
    ex.create()
    assert ex.has_remaining_files()
    (Path(ex.path) / "f.mp3").unlink()
    assert not ex.has_remaining_files()
    ex.cleanup()


@pytest.mark.parametrize("fmt", ["zip", "tar", "7z"])
def test_nested_and_empty_archive_remaining_file_signal(fmt, work):
    """Only real files count toward 'something left to import'.

    * a nested archive with empty dirs plus files reports remaining;
    * an archive holding nothing but empty directories reports none,
      matching what move-mode needs to decide complete vs partial.
    """
    p = work / f"nested{SUFFIX[fmt]}"
    MAKERS[fmt](
        p,
        STRUCTURES["nested"],
        (NESTED_EMPTY_DIR,),
    )
    with archives.ExtractedArchive(str(p), dir=work) as ex:
        assert ex.has_remaining_files()
        # Remove just one of the files -> partial import (archive kept).
        (Path(ex.path) / "a.mp3").unlink()
        assert ex.has_remaining_files()
        # Remove every remaining file (empty dirs linger) -> complete.
        for f in STRUCTURES["nested"]:
            target = Path(ex.path) / f
            if target.exists():
                target.unlink()
        assert not ex.has_remaining_files()

    p2 = work / f"onlyempty{SUFFIX[fmt]}"
    MAKERS[fmt](p2, {}, ("onlydir", "onlydir/nested"))
    with archives.ExtractedArchive(str(p2), dir=work) as ex:
        assert not ex.has_remaining_files()
        assert (Path(ex.path) / "onlydir" / "nested").is_dir()
