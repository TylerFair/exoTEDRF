"""Move reduction inputs and products to verified long-term storage."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from exotedrf.v2.config import _is_null_like as _null_like


def _digest(path, block=1 << 20):
    """Calculate a file SHA-256 digest in bounded blocks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(block), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root, checksums=True):
    """Describe file sizes, symlink targets and optional digests for a tree."""
    root = Path(root)
    if root.is_file() or root.is_symlink() and not root.is_dir():
        files = {'.': root}
    else:
        files = {}
        for directory, _, names in os.walk(root, followlinks=False):
            for name in names:
                path = Path(directory) / name
                files[os.fspath(path.relative_to(root))] = path
    manifest = {}
    for relative, path in files.items():
        if path.is_symlink():
            manifest[relative] = ('link', os.readlink(path))
        else:
            size = path.stat().st_size
            manifest[relative] = ('file', size, _digest(path) if checksums else None)
    return manifest


def safe_move(source, destination, *, checksums=True, logger=None):
    """Move a file or directory after verifying any cross-filesystem copy.

    An existing destination raises FileExistsError; a failed copy leaves the source intact.

    Parameters
    ----------
    source : str
        File or directory to move.
    destination : str
        Destination path, which must not already exist.
    checksums : bool
        If True, verify SHA-256 digests as well as file sizes.
    logger : None, callable
        Function to receive progress and warning messages.

    Returns
    -------
    destination : Path
        Verified destination path.
    """
    source = Path(os.path.expanduser(os.fspath(source)))
    destination = Path(os.path.expanduser(os.fspath(destination)))
    if not source.exists() and not source.is_symlink():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'{destination} already exists; refusing to merge or overwrite')
    if not destination.parent.is_dir():
        raise FileNotFoundError(f'archive destination {destination.parent} does not exist')
    try:
        os.rename(source, destination)
        return destination
    except OSError:
        pass

    temporary = destination.parent / (f'.{destination.name}.partial-{os.getpid()}')
    if temporary.exists():
        shutil.rmtree(temporary) if temporary.is_dir() else temporary.unlink()
    expected = _manifest(source, checksums=checksums)
    try:
        if source.is_dir() and not source.is_symlink():
            shutil.copytree(source, temporary, symlinks=True)
        else:
            shutil.copy2(source, temporary, follow_symlinks=False)
        copied = _manifest(temporary, checksums=checksums)
        if copied != expected:
            missing = sorted(set(expected) - set(copied))
            differing = sorted(key for key in expected if key in copied and copied[key] !=
                               expected[key])
            raise RuntimeError(f'archive verification failed for {source}: '
                f'{len(missing)} missing, {len(differing)} differing files')
        os.rename(temporary, destination)
    except BaseException:
        if temporary.exists() or temporary.is_symlink():
            if temporary.is_dir() and not temporary.is_symlink():
                shutil.rmtree(temporary, ignore_errors=True)
            else:
                temporary.unlink()
        raise
    # Remove the source after the verified copy is in place.
    if source.is_dir() and not source.is_symlink():
        shutil.rmtree(source)
    else:
        source.unlink()
    if logger is not None:
        logger(f'[v2] verified {len(expected)} file(s) at {destination}')
    return destination


def archive_run(cfg, output_root, *, logger=print, checksums=True):
    """Archive reduction inputs and outputs, reporting failures as warnings.

    Parameters
    ----------
    cfg : dict
        Reduction configuration.
    output_root : str
        Directory containing this reduction's products.
    logger : None, callable
        Function to receive progress and warning messages.
    checksums : bool
        If True, verify SHA-256 digests as well as file sizes.

    Returns
    -------
    archived : dict
        Destination paths for successfully archived inputs and outputs.
    """
    destination = cfg.get('archive_to_longterm_storage')
    if _null_like(destination):
        return {}
    destination = Path(os.path.expanduser(os.fspath(destination)))
    banner = '=' * 60

    def log(message):
        """Send an archive message to the configured logger."""
        if logger is not None:
            logger(message)

    log(f'\n{banner}\nARCHIVING TO LONG-TERM STORAGE\n{banner}\n')
    archived = {}
    if not destination.is_dir():
        # Require an existing archive destination.
        log(f'[v2] WARNING: archive destination {destination} does not exist; nothing was moved')
        return archived
    jobs = []
    input_dir = cfg.get('input_dir')
    if not _null_like(input_dir):
        jobs.append(('input', 'input data', Path(os.path.expanduser(os.fspath(input_dir)))))
    jobs.append(('output', 'pipeline outputs', Path(output_root)))
    for key, label, source in jobs:
        if not source.exists():
            log(f'[v2] WARNING: {label} directory not found (already moved?): {source}')
            continue
        target = destination / source.resolve().name
        log(f'Moving {label}:\n  From: {source}\n  To:   {target}')
        try:
            safe_move(source, target, checksums=checksums, logger=logger)
        except Exception as exc:
            log(f'  [v2] WARNING: failed to archive {label}: {exc}')
            continue
        archived[key] = str(target)
        log(f'  {label} archived successfully')
    log(f'\n{banner}\nARCHIVING COMPLETE\n{banner}\n')
    return archived


__all__ = ['archive_run', 'safe_move']
