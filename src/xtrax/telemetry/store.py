"""Content-addressed, compressed blob store for captured IR text.

Addressing is over the **uncompressed** bytes, which is what makes the store
useful: identical IR produced on two machines, by two runs, or by ten thousand
steps of the same shape signature collapses to one blob regardless of how it was
compressed. A digest recorded in a ledger row therefore stays verifiable by
anyone holding the original text, with no dependency on this module's
compression choices -- and changing the codec never invalidates an existing
store.

Sizing rationale (``scripts/measure_ir_compression.py``, 2026-08-28). On a
96-layer model, jaxpr + StableHLO + optimized HLO totalled 159 KB raw and 14 KB
gzipped -- an 11.3x ratio that *improves* with depth, because deep models emit
highly repetitive MLIR. IR size tracks op count, not batch size: a 16x larger
batch changed the bytes not at all. Storing full text is therefore cheap enough
that fingerprint-only capture is a fallback, not the default, and an embedded
database would add a dependency to solve a problem the measurements say does not
exist.

**Inode budget.** One blob is one file, so the population matters. It is bounded
by *distinct shape signatures*, not by runs or steps: 10,000 steps of one
compiled signature produce three blobs, and re-running unchanged code produces
zero new ones because the content -- and therefore the address -- is identical.
A year of daily runs against a stable model costs single-digit inodes. The way
that bound breaks is accidental per-step capture, which is why
:func:`check_population` exists and why callers must capture at the compile
boundary rather than inside a step.

**Codec.** gzip is the default because it is stdlib on this project's Python
floor. zstd would compress somewhat better, but ``compression.zstd`` is Python
3.14+ while xtrax requires >=3.13, so it would cost a third-party dependency to
save single-digit kilobytes per signature. The codec is nonetheless pluggable
and reads are suffix-agnostic, so a future switch (a 3.14 floor, or an installed
``zstandard``) neither orphans nor rewrites the blobs already on disk.
"""

import gzip
import hashlib
import os
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

_DIGEST_LEN = 64

# A population far above the distinct-shape-signature bound almost always means
# IR is being captured inside a step loop rather than at the compile boundary.
# Warning is the right response: the store still works, but the caller has lost
# the dedup property that keeps it cheap, and should know before it grows.
POPULATION_WARN_THRESHOLD = 10_000


def _gzip_compress(raw: bytes) -> bytes:
    # mtime=0 so the compressed bytes are a pure function of the content. gzip
    # embeds a timestamp by default, which would make two stores of identical IR
    # differ byte-for-byte and defeat any file-level reproducibility check.
    return gzip.compress(raw, compresslevel=6, mtime=0)


_CODECS: "dict[str, tuple[Callable[[bytes], bytes], Callable[[bytes], bytes]]]" = {
    ".gz": (_gzip_compress, gzip.decompress),
}

try:  # pragma: no cover - depends on an optional third-party package
    # Not a declared dependency: if a consumer happens to have it, blobs get a
    # better ratio; if not, gzip is used and existing blobs stay readable.
    import zstandard as _zstd  # ty: ignore[unresolved-import]

    _CODECS[".zst"] = (
        lambda raw: _zstd.ZstdCompressor(level=10).compress(raw),
        lambda blob: _zstd.ZstdDecompressor().decompress(blob),
    )
    PREFERRED_SUFFIX = ".zst"
except ImportError:
    PREFERRED_SUFFIX = ".gz"


class BlobStoreError(OSError):
    """Raised when the blob store cannot satisfy a read or write."""


def digest_of(text: str) -> "tuple[str, int]":
    """Return ``(sha256_hex, raw_byte_length)`` for ``text``.

    Split out from :meth:`BlobStore.put` so a caller can fingerprint an artifact
    it has decided *not* to store (the ``hash_only`` degradation mode) using
    exactly the same digest function, keeping the two paths comparable.
    """
    raw = text.encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), len(raw)


class BlobStore:
    """A directory of compressed blobs addressed by uncompressed sha256."""

    def __init__(self, root: "Path | str") -> None:
        self.root = Path(root)

    def _validate(self, sha256: str) -> None:
        if len(sha256) != _DIGEST_LEN:
            raise BlobStoreError(f"blob digest must be {_DIGEST_LEN} hex chars, got {sha256!r}")

    def path_for(self, sha256: str, suffix: str = PREFERRED_SUFFIX) -> Path:
        """Where a blob *would* be written under ``suffix``."""
        self._validate(sha256)
        return self.root / (sha256 + suffix)

    def find(self, sha256: str) -> "Path | None":
        """Locate an existing blob under any known codec suffix.

        Suffix-agnostic on purpose: the address is the uncompressed digest, so a
        blob written by gzip stays findable after the preferred codec changes.
        """
        self._validate(sha256)
        for suffix in _CODECS:
            candidate = self.root / (sha256 + suffix)
            if candidate.is_file():
                return candidate
        return None

    def has(self, sha256: str) -> bool:
        return self.find(sha256) is not None

    def put(self, text: str) -> "tuple[str, int]":
        """Store ``text``, returning ``(sha256, raw_byte_length)``.

        Idempotent, and idempotent *across codecs*: an artifact already present
        under any suffix is not rewritten. That is not merely an optimisation --
        rewriting would churn artifacts other ledger rows already reference, and
        would turn a re-run into a spurious filesystem change.

        The write is atomic (temp file + ``os.replace``), so a crash or a full
        disk mid-write cannot leave a truncated blob that a later reader would
        decompress into corrupt IR.
        """
        sha256, raw_len = digest_of(text)
        if self.find(sha256) is not None:
            return sha256, raw_len
        target = self.path_for(sha256)
        compress, _ = _CODECS[PREFERRED_SUFFIX]
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        try:
            tmp.write_bytes(compress(text.encode("utf-8")))
            os.replace(tmp, target)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise BlobStoreError(f"could not write blob {sha256}: {exc}") from exc
        return sha256, raw_len

    def get(self, sha256: str) -> str:
        """Read one blob back as text, decoding by its on-disk suffix."""
        path = self.find(sha256)
        if path is None:
            raise BlobStoreError(f"blob {sha256} is not present in {self.root}")
        _, decompress = _CODECS[path.suffix]
        try:
            return decompress(path.read_bytes()).decode("utf-8")
        except OSError as exc:
            raise BlobStoreError(f"could not read blob {sha256}: {exc}") from exc

    def verify(self, sha256: str) -> bool:
        """Whether the stored blob still hashes to the digest it is filed under.

        Cheap integrity check for a store meant to outlive the runs that wrote
        it; a silently corrupted blob would otherwise be indistinguishable from a
        genuine record at audit time.
        """
        if not self.has(sha256):
            return False
        actual, _ = digest_of(self.get(sha256))
        return actual == sha256

    def iter_digests(self) -> Iterator[str]:
        if not self.root.is_dir():
            return
        for path in sorted(self.root.iterdir()):
            if path.is_file() and path.suffix in _CODECS:
                yield path.name[: -len(path.suffix)]

    def count(self) -> int:
        return sum(1 for _ in self.iter_digests())

    def delete(self, sha256: str) -> bool:
        """Remove one blob under whichever codec holds it. Returns whether it existed."""
        path = self.find(sha256)
        if path is None:
            return False
        path.unlink()
        return True

    def total_bytes(self) -> int:
        """On-disk size of the store, compressed."""
        if not self.root.is_dir():
            return 0
        return sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())

    def check_population(self, threshold: int = POPULATION_WARN_THRESHOLD) -> int:
        """Warn if the blob population suggests IR is being captured per step.

        The store's cost model rests on one assumption: blobs are minted per
        distinct shape signature, so the count stays small no matter how long
        training runs. Capturing inside a step loop silently breaks that -- the
        store still functions, which is exactly why it needs to say something
        rather than quietly grow. Returns the observed count.
        """
        count = self.count()
        if count > threshold:
            warnings.warn(
                f"xtrax: the IR blob store at {self.root} holds {count:,} blobs, "
                f"above the {threshold:,} expected from distinct shape signatures. "
                "This usually means IR is being captured inside a step loop rather "
                "than at the compile boundary; run `xtrax ledger compact` to garbage-"
                "collect unreferenced blobs and check the capture call site.",
                UserWarning,
                stacklevel=2,
            )
        return count
