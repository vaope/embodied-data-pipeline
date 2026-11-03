"""Selective, single-episode download from the public EgoDex zip archives.

implementation-plan.md section 4 says to use the official public EgoDex *test*
set and to *not* download the full training set.  The published test archive is
a single 16 GB zip containing ~6600 paired ``*.hdf5`` / ``*.mp4`` episodes, so
downloading it whole is wasteful when the first closed loop needs one short
clip.

This module therefore reads the archive's central directory over HTTP range
requests and extracts only the two members of one episode.  Nothing here is a
reimplementation of EgoDex: it is a transport helper for
:class:`~ego2robot.ingest.egodex.EgoDexHandAdapter`.

Source of truth for the layout:
https://github.com/apple/ml-egodex (README, section "Dataset Structure").

Dataset licence: CC-BY-NC-ND (see the EgoDex repository).  Nothing downloaded
by this module may be redistributed; the resulting episode is marked
``publishable: false``.
"""

from __future__ import annotations

import struct
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

__all__ = [
    "EGODEX_ARCHIVES",
    "EgoDexArchive",
    "EgoDexEpisodeRef",
    "EgoDexZipIndex",
    "download_episode",
    "humanize_bytes",
    "open_archive",
]

EGODEX_ARCHIVES: dict[str, str] = {
    "test": "https://ml-site.cdn-apple.com/datasets/egodex/test.zip",
    "extra": "https://ml-site.cdn-apple.com/datasets/egodex/extra.zip",
    "part1": "https://ml-site.cdn-apple.com/datasets/egodex/part1.zip",
    "part2": "https://ml-site.cdn-apple.com/datasets/egodex/part2.zip",
    "part3": "https://ml-site.cdn-apple.com/datasets/egodex/part3.zip",
    "part4": "https://ml-site.cdn-apple.com/datasets/egodex/part4.zip",
    "part5": "https://ml-site.cdn-apple.com/datasets/egodex/part5.zip",
}

_UA = {"User-Agent": "ego2robot/0.1 (single-episode EgoDex fetch)"}
_CHUNK = 1 << 20


def humanize_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


@dataclass(frozen=True)
class EgoDexEpisodeRef:
    """One episode inside an EgoDex archive."""

    task: str
    index: int
    hdf5_member: str
    mp4_member: str
    hdf5_size: int
    mp4_size: int

    @property
    def name(self) -> str:
        return f"{self.task}/{self.index}"

    @property
    def total_size(self) -> int:
        return self.hdf5_size + self.mp4_size

    def describe(self) -> str:
        return (
            f"{self.name}  hdf5={humanize_bytes(self.hdf5_size)} "
            f"mp4={humanize_bytes(self.mp4_size)} total={humanize_bytes(self.total_size)}"
        )


@dataclass
class EgoDexArchive:
    """Remote EgoDex archive addressable by byte range."""

    name: str
    url: str
    size: int = 0
    supports_range: bool = False

    @property
    def is_remote(self) -> bool:
        return self.url.startswith(("http://", "https://"))


def _head(url: str, *, timeout: float = 60.0) -> tuple[int, bool]:
    req = urllib.request.Request(url, headers={**_UA, "Range": "bytes=0-0"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = resp.status
        cr = resp.headers.get("Content-Range", "")
        total = 0
        if "/" in cr:
            total = int(cr.rsplit("/", 1)[1])
        elif resp.headers.get("Content-Length"):
            total = int(resp.headers["Content-Length"])
        return total, status == 206


def open_archive(name: str = "test", *, url: str | None = None) -> EgoDexArchive:
    """Resolve an archive name to a range-capable remote (or local) archive."""
    if url is None:
        if name not in EGODEX_ARCHIVES:
            raise KeyError(f"unknown EgoDex archive {name!r}; known: {sorted(EGODEX_ARCHIVES)}")
        url = EGODEX_ARCHIVES[name]
    if url.startswith(("http://", "https://")):
        size, ranged = _head(url)
        return EgoDexArchive(name=name, url=url, size=size, supports_range=ranged)
    p = Path(url)
    if not p.is_file():
        raise FileNotFoundError(f"local archive not found: {p}")
    return EgoDexArchive(name=name, url=str(p), size=p.stat().st_size, supports_range=True)


class EgoDexZipIndex:
    """Random-access reader over a (possibly ZIP64, possibly remote) zip.

    Only the bytes needed for one member are ever transferred: the central
    directory is read in one range request, and member data is streamed with
    ``Range`` requests.
    """

    def __init__(self, archive: EgoDexArchive, *, timeout: float = 120.0) -> None:
        self.archive = archive
        self.timeout = timeout
        self.entries: dict[str, tuple[int, int, int, int]] = {}
        # member -> (local_header_offset, compressed_size, uncompressed_size, method)
        self._load_central_directory()

    # -- transport ---------------------------------------------------------

    def _read_range(self, start: int, end: int) -> bytes:
        """Read ``[start, end]`` inclusive."""
        if start < 0:
            start = max(0, self.archive.size + start)
        if not self.archive.is_remote:
            with open(self.archive.url, "rb") as fh:
                fh.seek(start)
                return fh.read(end - start + 1)
        req = urllib.request.Request(
            self.archive.url, headers={**_UA, "Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()

    # -- central directory -------------------------------------------------

    def _load_central_directory(self) -> None:
        size = self.archive.size
        if size == 0:
            raise ValueError(f"{self.archive.name}: archive size is unknown")
        tail = self._read_range(max(0, size - 70000), size - 1)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise ValueError(f"{self.archive.name}: no end-of-central-directory record found")
        _sig, _d, _cd, _nt, n_total, cd_size, cd_off, _cl = struct.unpack("<IHHHHIIH", tail[i : i + 22])

        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n_total == 0xFFFF:
            j = tail.rfind(b"PK\x06\x07")
            if j < 0:
                raise ValueError(
                    f"{self.archive.name}: ZIP64 markers present but no ZIP64 EOCD locator found"
                )
            _s, _disk, z64_off, _nd = struct.unpack("<IIQI", tail[j : j + 20])
            z = self._read_range(z64_off, z64_off + 55)
            (_s2, _sz, _vm, _vn, _d1, _d2, _n1, n_total, cd_size, cd_off) = struct.unpack(
                "<IQHHIIQQQQ", z[:56]
            )

        cd = self._read_range(cd_off, cd_off + cd_size - 1)
        self._parse_central_directory(cd, n_total)

    def _parse_central_directory(self, cd: bytes, n_expected: int) -> None:
        pos = 0
        count = 0
        while pos + 46 <= len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
            (
                _sig,
                _vmade,
                _vneed,
                _flags,
                method,
                _mtime,
                _mdate,
                _crc,
                csize,
                usize,
                _nlen,
                _elen,
                _clen,
                _dstart,
                _iattr,
                _eattr,
                lho,
            ) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos : pos + 46])
            name_len, extra_len, comment_len = struct.unpack("<HHH", cd[pos + 28 : pos + 34])
            name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8", "replace")
            extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]

            if lho == 0xFFFFFFFF or csize == 0xFFFFFFFF or usize == 0xFFFFFFFF:
                lho, csize, usize = _apply_zip64_extra(extra, lho, csize, usize)

            self.entries[name] = (lho, csize, usize, method)
            pos += 46 + name_len + extra_len + comment_len
            count += 1
        if count != n_expected:
            raise ValueError(
                f"{self.archive.name}: parsed {count} central directory entries, "
                f"header declared {n_expected}"
            )

    # -- queries -----------------------------------------------------------

    def episodes(self, *, tasks: Sequence[str] | None = None) -> list[EgoDexEpisodeRef]:
        """Enumerate paired ``task/index.hdf5`` + ``task/index.mp4`` episodes."""
        by_key: dict[tuple[str, int], dict[str, str]] = {}
        for name in self.entries:
            if name.endswith("/"):
                continue
            parts = name.split("/")
            if len(parts) < 2:
                continue
            stem, ext = parts[-1].rsplit(".", 1) if "." in parts[-1] else (parts[-1], "")
            if ext not in {"hdf5", "mp4"} or not stem.isdigit():
                continue
            task = "/".join(parts[:-1])
            if tasks is not None and task not in tasks:
                continue
            by_key.setdefault((task, int(stem)), {})[ext] = name

        refs: list[EgoDexEpisodeRef] = []
        for (task, index), members in sorted(by_key.items()):
            if "hdf5" not in members or "mp4" not in members:
                continue
            h, m = members["hdf5"], members["mp4"]
            refs.append(
                EgoDexEpisodeRef(
                    task=task,
                    index=index,
                    hdf5_member=h,
                    mp4_member=m,
                    hdf5_size=self.entries[h][2],
                    mp4_size=self.entries[m][2],
                )
            )
        return refs

    def task_names(self) -> list[str]:
        return sorted({r.task for r in self.episodes()})

    def read_member(self, member: str, *, progress: Iterable[int] | None = None) -> bytes:
        """Fetch and decompress one member into memory."""
        if member not in self.entries:
            raise KeyError(f"{self.archive.name}: member {member!r} not in the central directory")
        lho, csize, usize, method = self.entries[member]
        header = self._read_range(lho, lho + 29)
        if header[:4] != b"PK\x03\x04":
            raise ValueError(f"{member}: bad local file header magic {header[:4]!r}")
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        data_start = lho + 30 + name_len + extra_len
        if method == 0:
            raw = self._read_range(data_start, data_start + csize - 1)
            return raw
        if method != 8:
            raise ValueError(f"{member}: unsupported compression method {method}")
        import zlib

        out = bytearray()
        remaining = csize
        offset = data_start
        decomp = zlib.decompressobj(-zlib.MAX_WBITS)
        while remaining > 0:
            block = min(_CHUNK, remaining)
            chunk = self._read_range(offset, offset + block - 1)
            offset += block
            remaining -= block
            out.extend(decomp.decompress(chunk))
        out.extend(decomp.flush())
        if len(out) != usize:
            raise ValueError(f"{member}: decompressed {len(out)} bytes, expected {usize}")
        return bytes(out)

    def extract_member(self, member: str, dest: str | Path) -> Path:
        """Fetch one member and write it atomically to ``dest``."""
        from ..core.artifacts import atomic_write_bytes

        return atomic_write_bytes(dest, self.read_member(member))


def _apply_zip64_extra(extra: bytes, lho: int, csize: int, usize: int) -> tuple[int, int, int]:
    """Resolve 0xFFFFFFFF placeholders from the ZIP64 extended information field."""
    pos = 0
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack("<HH", extra[pos : pos + 4])
        body = extra[pos + 4 : pos + 4 + data_size]
        if header_id == 0x0001:
            cursor = 0
            if usize == 0xFFFFFFFF:
                usize = struct.unpack("<Q", body[cursor : cursor + 8])[0]
                cursor += 8
            if csize == 0xFFFFFFFF:
                csize = struct.unpack("<Q", body[cursor : cursor + 8])[0]
                cursor += 8
            if lho == 0xFFFFFFFF:
                lho = struct.unpack("<Q", body[cursor : cursor + 8])[0]
                cursor += 8
            break
        pos += 4 + data_size
    return lho, csize, usize


# ---------------------------------------------------------------------------
# Local extraction (already-downloaded zips)
# ---------------------------------------------------------------------------


def extract_local_episode(zip_path: str | Path, ref: EgoDexEpisodeRef, dest_dir: str | Path) -> tuple[Path, Path]:
    """Extract an episode from a zip that is already on disk."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in (ref.hdf5_member, ref.mp4_member):
            target = dest / Path(member).name
            with zf.open(member) as src, open(target, "wb") as out:
                while True:
                    block = src.read(_CHUNK)
                    if not block:
                        break
                    out.write(block)
    return dest / Path(ref.hdf5_member).name, dest / Path(ref.mp4_member).name


def download_episode(
    ref: EgoDexEpisodeRef,
    index: EgoDexZipIndex,
    dest_dir: str | Path,
    *,
    log: object | None = None,
) -> tuple[Path, Path]:
    """Fetch both members of one episode into ``dest_dir`` (flat file names)."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    hdf5_path = dest / Path(ref.hdf5_member).name
    mp4_path = dest / Path(ref.mp4_member).name
    index.extract_member(ref.hdf5_member, hdf5_path)
    index.extract_member(ref.mp4_member, mp4_path)
    if log is not None and hasattr(log, "info"):
        log.info(f"fetched {ref.name}: {hdf5_path.name} + {mp4_path.name}")
    return hdf5_path, mp4_path


def select_episode(
    refs: Sequence[EgoDexEpisodeRef],
    *,
    task: str | None = None,
    index: int | None = None,
    max_total_bytes: int = 80 * 1024 * 1024,
) -> EgoDexEpisodeRef:
    """Pick one episode deterministically.

    Default policy follows implementation-plan.md section 4 ("5-10 s, single
    hand, small camera motion"): among the small episodes of the requested task
    the lowest index wins, so a rerun selects the same episode.
    """
    pool = [r for r in refs if task is None or r.task == task]
    if index is not None:
        pool = [r for r in pool if r.index == index]
    if not pool:
        raise LookupError(
            f"no EgoDex episode matched task={task!r} index={index!r}; "
            f"available tasks: {len({r.task for r in refs})}"
        )
    small = [r for r in pool if r.total_size <= max_total_bytes]
    return min(small or pool, key=lambda r: (r.total_size, r.task, r.index))


def summarise_sizes(refs: Sequence[EgoDexEpisodeRef], *, limit: int = 10) -> list[str]:
    """Human-readable size lines, smallest first."""
    ordered = sorted(refs, key=lambda r: r.total_size)
    return [r.describe() for r in ordered[:limit]]


def _thin(array: np.ndarray) -> np.ndarray:  # pragma: no cover - helper for callers
    return np.asarray(array, dtype=np.float32)


def _iter_episode_names(entries: Iterable[str]) -> Iterator[tuple[str, int]]:
    for name in entries:
        if not name.endswith(".hdf5"):
            continue
        parts = name.split("/")
        stem = parts[-1][:-5]
        if stem.isdigit():
            yield "/".join(parts[:-1]), int(stem)
