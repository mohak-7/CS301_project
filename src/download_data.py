"""Download the schema and question files needed by the study (about 12 MB in total).

Implements Phase 3 data acquisition (implementation_plan.md, §3.2, D15). Database files are
never downloaded; from the Bird archives only the needed JSON members are extracted.
"""

import argparse
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import nltk
from tqdm import tqdm

from src.common import PROJECT_ROOT, load_config, setup_logging

logger = logging.getLogger(__name__)
USER_AGENT = "cs301-schema-routing/0.1"


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable file view of a remote file, fetched with HTTP range requests.

    `zipfile` only reads the archive index and the requested members, so wrapping a remote zip
    in this class transfers a few hundred KB instead of the whole archive.
    """

    def __init__(self, url: str) -> None:
        """Open the remote file and read its size.

        Args:
            url: HTTP(S) URL of a server that supports `Range` requests.
        """
        super().__init__()
        self.url = url
        self.position = 0
        self.bytes_fetched = 0
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        """Report that the file can be read."""
        return True

    def seekable(self) -> bool:
        """Report that the file supports random access."""
        return True

    def tell(self) -> int:
        """Return the current byte offset."""
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move the current byte offset.

        Args:
            offset: Byte offset relative to `whence`.
            whence: `io.SEEK_SET`, `io.SEEK_CUR` or `io.SEEK_END`.

        Returns:
            The new absolute offset.
        """
        origins = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        self.position = origins[whence] + offset
        return self.position

    def readinto(self, buffer: bytearray | memoryview) -> int:
        """Fill `buffer` with bytes starting at the current offset.

        Args:
            buffer: Writable buffer; its length is the number of bytes requested.

        Returns:
            Number of bytes actually read (0 at end of file).
        """
        end = min(self.size, self.position + len(buffer))
        if end <= self.position:
            return 0
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self.position}-{end - 1}", "User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.position += len(data)
        self.bytes_fetched += len(data)
        return len(data)


def download_file(url: str, target: Path, force: bool = False) -> None:
    """Download one file over HTTP(S).

    Args:
        url: Source URL.
        target: Destination path.
        force: Re-download even if the target already exists.
    """
    if target.exists() and not force:
        logger.info("skip (exists): %s", target.relative_to(PROJECT_ROOT))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        target.write_bytes(response.read())
    logger.info(
        "downloaded %s (%.1f KB)", target.relative_to(PROJECT_ROOT), target.stat().st_size / 1e3
    )


def extract_zip_members(
    url: str, members: dict[str, str], raw_dir: Path, force: bool = False
) -> None:
    """Extract selected members of a remote zip without downloading the whole archive.

    Args:
        url: URL of the zip archive (server must support HTTP range requests).
        members: Mapping from archive member path to destination path relative to `raw_dir`.
        raw_dir: Root directory for raw data.
        force: Re-extract even if all targets already exist.
    """
    targets = {member: raw_dir / dest for member, dest in members.items()}
    if not force and all(path.exists() for path in targets.values()):
        logger.info("skip (exists): members of %s", url)
        return
    remote = HttpRangeFile(url)
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20)) as archive:
        for member, path in targets.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(member))
            logger.info("extracted %s -> %s", member, path.relative_to(PROJECT_ROOT))
    logger.info(
        "fetched %.1f KB of a %.1f MB archive", remote.bytes_fetched / 1e3, remote.size / 1e6
    )


def download_tokenizer_data(nltk_dir: Path) -> None:
    """Download nltk's `punkt_tab` tokenizer data into the project.

    The Spider hardness code (third_party/test_suite_sql_eval/process_sql.py) tokenises SQL
    with `nltk.word_tokenize`, which needs this data (§3.5).

    Args:
        nltk_dir: Project-local nltk data directory.
    """
    nltk_dir.mkdir(parents=True, exist_ok=True)
    nltk.download("punkt_tab", download_dir=str(nltk_dir), quiet=True)
    logger.info("nltk punkt_tab available in %s", nltk_dir.relative_to(PROJECT_ROOT))


def main() -> None:
    """Download every source listed in configs/data.yaml."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml", help="data source config")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    raw_dir = PROJECT_ROOT / config["raw_dir"]

    for source in tqdm(config["downloads"], desc="files"):
        # Third-party code goes to its own folder, apart from the data (§S.3).
        root = PROJECT_ROOT / source.get("root", config["raw_dir"])
        download_file(source["url"], root / source["target"], force=args.force)
    download_tokenizer_data(PROJECT_ROOT / config["nltk_dir"])
    for archive in config["zip_members"]:
        extract_zip_members(archive["url"], archive["members"], raw_dir, force=args.force)


if __name__ == "__main__":
    main()
