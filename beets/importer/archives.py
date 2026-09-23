"""Uniform handling for the archive formats the importer can extract.

`zip` and `tar` are backed by the standard library; `rar` and `7z`
need the optional `rarfile` and `py7zr` packages. Every format is
described by an `ArchiveType` providing:

* `sniff` -- cheaply test whether a path is an archive of this type
  (magic-byte based, so this does not require the optional package),
* `open_archive` -- a context-manager factory returning an
  `Archive` for member iteration and extraction.

All members (files and directories) are iterated once, in the format's
on-disk order, through the common `ArchiveMember` interface, so callers
no longer need format-specific traversal logic.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from zipfile import ZipFile, is_zipfile

StrPath = str | os.PathLike[str]


class ArchiveError(Exception):
    """Base class for errors while opening or extracting an archive."""


class ArchiveDependencyError(ArchiveError):
    """An archive needs an optional dependency that is not installed.

    Raised when the file itself is recognised (via magic bytes) but the
    package required to read it is unavailable, so the failure has a
    clear, actionable cause instead of an opaque extraction error.
    """


@dataclass(frozen=True)
class ArchiveMember:
    """A single entry of an archive in a format-independent shape.

    `name` is the archive-relative path, using the archive's own
    (POSIX-style) separators. `is_dir` marks directory entries.
    `mtime` is the member modification time (seconds since the epoch)
    when the format records one, else None.
    """

    name: str
    is_dir: bool = False
    mtime: float | None = None


class Archive:
    """Common archive interface: iterate members and extract them.

    Subclasses adapt a concrete library object (`ZipFile`, `TarFile`,
    ...) to this interface. Instances are used as context managers and
    must guarantee the underlying file is closed on normal exit *and* on
    errors during extraction.
    """

    def __enter__(self) -> Archive:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def members(self) -> Iterator[ArchiveMember]:
        """Yield every member (files and directories) exactly once."""
        raise NotImplementedError

    def extractall(self, path: str) -> None:
        """Extract every member beneath the existing directory `path`."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class ZipArchive(Archive):
    def __init__(self, path: str) -> None:
        self._zip = ZipFile(path)

    def members(self) -> Iterator[ArchiveMember]:
        for info in self._zip.infolist():
            yield ArchiveMember(
                name=info.filename,
                # Central-flag directory entries always end with '/'.
                is_dir=info.is_dir(),
                mtime=time.mktime((*info.date_time, 0, 0, -1)),
            )

    def extractall(self, path: str) -> None:
        self._zip.extractall(path)

    def close(self) -> None:
        self._zip.close()


class TarArchive(Archive):
    def __init__(self, path: str) -> None:
        # ``set_info`` restores member mtimes/modes during extraction, so
        # no post-extraction walk (a la zip) is needed.
        self._tar = tarfile.open(path)

    def members(self) -> Iterator[ArchiveMember]:
        for info in self._tar.getmembers():
            # An empty arcname can yield a meaningless root entry.
            if not info.name:
                continue
            yield ArchiveMember(
                name=info.name,
                is_dir=info.isdir(),
                mtime=float(info.mtime),
            )

    def extractall(self, path: str) -> None:
        self._tar.extractall(path)

    def close(self) -> None:
        self._tar.close()


@dataclass(frozen=True)
class ArchiveType:
    """Description of one supported archive format.

    `sniff` recognises the format from a path (usually by magic bytes),
    and `open_archive` returns a context manager yielding the `Archive`.
    `dependency` names the optional package, if any.
    """

    name: str
    extensions: tuple[str, ...]
    dependency: str | None
    sniff: object
    open_archive: object

    def is_supported(self) -> bool:
        return self.dependency is None or _module_available(self.dependency)


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _open_rar(path: str) -> Archive:
    try:
        from rarfile import RarFile
    except ImportError:
        raise ArchiveDependencyError(
            "extracting RAR archives requires the 'rarfile' package"
        )
    return _RarArchive(RarFile(path))


class _RarArchive(Archive):
    def __init__(self, rar: object) -> None:
        self._rar = rar

    def members(self) -> Iterator[ArchiveMember]:
        # ``infolist()`` covers files and directories; ``namelist()``
        # omits directories and so would lose empty ones.
        for info in self._rar.infolist():
            yield ArchiveMember(
                name=info.filename,
                is_dir=getattr(info, "is_dir", lambda: False)(),
                mtime=time.mktime((*info.date_time, 0, 0, -1)),
            )

    def extractall(self, path: str) -> None:
        # rarfile restores stored mtimes while extracting.
        self._rar.extractall(path)

    def close(self) -> None:
        self._rar.close()


def _is_rarfile(path: str) -> bool:
    try:
        from rarfile import is_rarfile
    except ImportError:
        # Recognised by extension below; reading still raises a clear
        # ArchiveDependencyError.
        return False
    return is_rarfile(path)


def _open_7z(path: str) -> Archive:
    try:
        from py7zr import SevenZipFile
    except ImportError:
        raise ArchiveDependencyError(
            "extracting 7z archives requires the 'py7zr' package"
        )
    return _SevenZipArchive(SevenZipFile(path, mode="r"))


class _SevenZipArchive(Archive):
    def __init__(self, sevenzip: object) -> None:
        self._7z = sevenzip

    def members(self) -> Iterator[ArchiveMember]:
        # ``list()`` includes directory entries, unlike ``namelist()``.
        for info in self._7z.list():
            if not info.filename:
                continue
            last_write = getattr(info, "lastwritetime", None)
            mtime = float(last_write.totimestamp()) if last_write else None
            yield ArchiveMember(
                name=info.filename,
                is_dir=bool(info.is_directory),
                mtime=mtime,
            )

    def extractall(self, path: str) -> None:
        # py7zr restores stored mtimes while extracting.
        self._7z.extractall(path)

    def close(self) -> None:
        self._7z.close()


def _is_7zfile(path: str) -> bool:
    try:
        from py7zr import is_7zfile
    except ImportError:
        return False
    return is_7zfile(path)


# All formats, including those whose optional dependency is missing:
# detection stays cheap (magic bytes/extension) and opening produces a
# clear ArchiveDependencyError.
ARCHIVE_TYPES: tuple[ArchiveType, ...] = (
    ArchiveType(
        name="zip",
        extensions=(".zip",),
        dependency=None,
        sniff=is_zipfile,
        open_archive=ZipArchive,
    ),
    ArchiveType(
        name="tar",
        extensions=(".tar",),
        dependency=None,
        sniff=tarfile.is_tarfile,
        open_archive=TarArchive,
    ),
    ArchiveType(
        name="rar",
        extensions=(".rar",),
        dependency="rarfile",
        sniff=_is_rarfile,
        open_archive=_open_rar,
    ),
    ArchiveType(
        name="7z",
        extensions=(".7z",),
        dependency="py7zr",
        sniff=_is_7zfile,
        open_archive=_open_7z,
    ),
)


def is_archive(path: StrPath) -> bool:
    """Return True if `path` is a file in a recognised archive format.

    Detection prefers magic bytes via the format's own sniffer. The
    extension is used as a fallback only for formats whose reader
    package is not installed: a ``.rar``/``.7z`` file is then still
    recognised (and opening it later raises a clear
    `ArchiveDependencyError`), while a corrupt file whose reader *is*
    available (e.g. a broken ``.zip``) is not misreported as an archive.
    """
    if not os.path.isfile(path):
        return False

    name = os.fsdecode(path)
    lower = name.lower()
    for archive_type in ARCHIVE_TYPES:
        if archive_type.sniff(name):
            return True
    for archive_type in ARCHIVE_TYPES:
        if (
            archive_type.extensions
            and lower.endswith(archive_type.extensions)
            and not archive_type.is_supported()
        ):
            return True
    return False


def get_archive_type(path: StrPath) -> ArchiveType:
    """Return the `ArchiveType` matching `path`.

    Raises `ArchiveError` when no format matches, and
    `ArchiveDependencyError` when the only matching format needs an
    optional package that is not installed.
    """
    name = os.fsdecode(path)
    fallback: ArchiveType | None = None
    for archive_type in ARCHIVE_TYPES:
        if archive_type.sniff(name):
            if not archive_type.is_supported():
                raise ArchiveDependencyError(
                    f"extracting {archive_type.name} archives requires the "
                    f"'{archive_type.dependency}' package"
                )
            return archive_type
        if fallback is None and name.lower().endswith(archive_type.extensions):
            fallback = archive_type

    if fallback is not None:
        if not fallback.is_supported():
            raise ArchiveDependencyError(
                f"extracting {fallback.name} archives requires the "
                f"'{fallback.dependency}' package"
            )
        # E.g. a mislabelled ".rar" the rarfile reader does not accept;
        # let the reader surface the concrete corruption error.
        return fallback

    raise ArchiveError(f"no handler found for archive: {name}")


@contextmanager
def open_archive(path: StrPath) -> Iterator[Archive]:
    """Open the archive at `path` as a context-managed `Archive`.

    The underlying file is always closed, including when extraction or
    member iteration raises mid-way.
    """
    archive_type = get_archive_type(path)
    try:
        archive = archive_type.open_archive(os.fsdecode(path))
    except ArchiveError:
        raise
    except Exception as exc:
        # Corrupt/unreadable archive: surface a uniform error type.
        raise ArchiveError(
            f"could not open archive {path}: {exc}"
        ) from exc
    try:
        yield archive
    finally:
        archive.close()


def restore_mtimes(archive: Archive, extract_to: str) -> None:
    """Set extracted files' mtimes from the archive members.

    Iterating the common `ArchiveMember` sequence (instead of a
    format-specific ``infolist()``) makes this work identically for zip,
    tar, rar and 7z. Members without a recorded time (e.g. 7z entries
    lacking a last-write timestamp) are left untouched.
    """
    for member in archive.members():
        if member.mtime is None:
            continue
        fullpath = os.path.join(extract_to, member.name)
        # Entries may be missing for exotic members (symlinks whose
        # target was not extracted, etc.); skip rather than abort.
        if not os.path.lexists(fullpath):
            continue
        try:
            os.utime(fullpath, (member.mtime, member.mtime), follow_symlinks=False)
        except (NotImplementedError, OSError):
            os.utime(fullpath, (member.mtime, member.mtime))


class ExtractedArchive:
    """Lifecycle of an archive extracted into a temporary directory.

    One object per extraction. `create()` extracts the archive at
    `archive_path` into a fresh temporary directory; `cleanup()` removes
    that directory. The temporary directory is created *before* opening
    the archive and removed on every failure path, so a corrupt or empty
    archive, a missing optional dependency, or an error mid-extraction
    never leaks temporary files.

    Instances are reusable as context managers::

        with ExtractedArchive(archive_path) as extracted:
            ... work in extracted.path ...

    On context exit (or via `cleanup()`) the directory is always
    removed; the source archive itself is left for the caller to manage.
    """

    def __init__(
        self,
        archive_path: StrPath,
        *,
        prefix: str = "beets-",
        dir: StrPath | None = None,
    ) -> None:
        self.archive_path = archive_path
        self._prefix = prefix
        # Parent directory for the temporary directory (None: the
        # system default). Exposed mainly so tests can observe cleanup.
        self._dir = os.fspath(dir) if dir is not None else None
        self.path: str | None = None

    def create(self) -> str:
        """Extract the archive into a new temporary directory.

        Returns that directory's path (also available as `self.path`).
        Raises `ArchiveError`/`ArchiveDependencyError` or the underlying
        extraction error; in every failure case the temporary directory
        is rolled back and `self.path` stays None.
        """
        extract_to = tempfile.mkdtemp(prefix=self._prefix, dir=self._dir)
        try:
            with open_archive(self.archive_path) as archive:
                archive.extractall(extract_to)
                restore_mtimes(archive, extract_to)
        except BaseException:
            # Roll back the partial extraction so no temp directory leaks.
            shutil.rmtree(extract_to, ignore_errors=True)
            raise
        self.path = extract_to
        return extract_to

    def __enter__(self) -> ExtractedArchive:
        self.create()
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def is_extracted(self) -> bool:
        return self.path is not None

    def has_remaining_files(self) -> bool:
        """Return True if the extraction directory still contains files.

        Used in move mode to tell a complete import (every extracted
        file moved away) from a partial one.
        """
        assert self.path is not None
        return any(files for _, _, files in os.walk(self.path))

    def cleanup(self) -> None:
        """Remove the temporary extraction directory.

        The source archive itself is never touched here; callers decide
        (e.g. in move mode after every member was imported) whether to
        delete it separately. Safe to call more than once or after a
        failed extraction: without a live extraction directory this is a
        no-op.
        """
        if self.path is None:
            return

        extract_to = self.path
        self.path = None
        shutil.rmtree(extract_to, ignore_errors=True)
