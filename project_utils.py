"""Local file utilities. No downloads, model loading, or training on import."""

from pathlib import Path, PurePosixPath
import json
import os
import stat
import tempfile
import zipfile


def project_path(relative):
    """Resolve a project-relative path, optionally under PROJECT_DATA_DIR."""
    root = Path(os.environ.get("PROJECT_DATA_DIR", Path(__file__).resolve().parent)).expanduser()
    return str(root / relative)


def save_checkpoint(checkpoint_file, data):
    """Atomically replace JSON so an interrupted write cannot corrupt a checkpoint."""
    path = Path(checkpoint_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(checkpoint_file):
    path = Path(checkpoint_file)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object checkpoint")
    return data


def unzip_data(filename, destination=".", max_bytes=2_000_000_000, max_members=100_000):
    """Extract a ZIP after rejecting traversal, links, overwrite, and oversized data.

    The destination must be a local directory not writable by untrusted users.
    Limits apply to both declared sizes and actual streamed bytes.
    """
    root = Path(destination).resolve()
    with zipfile.ZipFile(filename) as archive:
        members = archive.infolist()
        if len(members) > max_members or sum(m.file_size for m in members) > max_bytes:
            raise ValueError("Archive exceeds extraction limits")
        targets = set()
        for member in members:
            name = member.filename
            parts = PurePosixPath(name).parts
            if (
                not parts
                or name.startswith(("/", "\\"))
                or "\\" in name
                or ":" in parts[0]
                or ".." in parts
            ):
                raise ValueError("Unsafe archive path")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Archive symlinks are not allowed")
            target = (root / name).resolve()
            if not target.is_relative_to(root) or target in targets:
                raise ValueError("Archive path escapes destination or is duplicated")
            if target.exists() and not (member.is_dir() and target.is_dir()):
                raise FileExistsError(target)
            targets.add(target)
        root.mkdir(parents=True, exist_ok=True)
        total = 0
        for member in members:
            target = root / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("Expanded archive exceeds extraction limit")
                    output.write(chunk)
    return root
