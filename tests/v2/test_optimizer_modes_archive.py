"""archive_to_longterm_storage: v1 move semantics, never losing data."""

import os

import pytest

from exotedrf.v2 import archive


def _tree(root, files):
    """Return tree."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_same_filesystem_move_is_a_rename(tmp_path):
    """Check same filesystem move is a rename."""
    source = _tree(tmp_path / 'run', {'a.txt': 'a', 'sub/b.txt': 'b'})
    (tmp_path / 'dest').mkdir()
    target = archive.safe_move(source, tmp_path / 'dest' / 'run')
    assert not source.exists()
    assert (target / 'sub' / 'b.txt').read_text() == 'b'


def _cross_device(monkeypatch):
    """Return cross device."""
    real = os.rename
    calls = {'n': 0}

    def rename(src, dst):
        calls['n'] += 1
        if calls['n'] == 1:
            raise OSError(18, 'Invalid cross-device link')
        return real(src, dst)

    monkeypatch.setattr(archive.os, 'rename', rename)


def test_cross_device_move_copies_verifies_then_deletes(tmp_path,
                                                         monkeypatch):
    """Check cross device move copies verifies then deletes."""
    source = _tree(tmp_path / 'run', {'a.txt': 'a' * 1000,
                                      'sub/b.fits': 'b'})
    os.symlink('a.txt', source / 'link')
    (tmp_path / 'dest').mkdir()
    _cross_device(monkeypatch)
    target = archive.safe_move(source, tmp_path / 'dest' / 'run')
    assert not source.exists()
    assert (target / 'a.txt').read_text() == 'a' * 1000
    assert os.readlink(target / 'link') == 'a.txt'
    assert not any(p.name.startswith('.run.partial')
                   for p in (tmp_path / 'dest').iterdir())


def test_failed_verification_keeps_the_source(tmp_path, monkeypatch):
    """Check failed verification keeps the source."""
    source = _tree(tmp_path / 'run', {'a.txt': 'a'})
    (tmp_path / 'dest').mkdir()
    _cross_device(monkeypatch)
    real_copytree = archive.shutil.copytree

    def corrupting_copytree(src, dst, **kwargs):
        real_copytree(src, dst, **kwargs)
        (dst / 'a.txt').write_text('corrupted')

    monkeypatch.setattr(archive.shutil, 'copytree', corrupting_copytree)
    with pytest.raises(RuntimeError, match='verification failed'):
        archive.safe_move(source, tmp_path / 'dest' / 'run')
    assert (source / 'a.txt').read_text() == 'a'
    assert list((tmp_path / 'dest').iterdir()) == []


def test_existing_destination_is_never_merged(tmp_path):
    """Check existing destination is never merged."""
    source = _tree(tmp_path / 'run', {'a.txt': 'new'})
    _tree(tmp_path / 'dest' / 'run', {'a.txt': 'old'})
    with pytest.raises(FileExistsError):
        archive.safe_move(source, tmp_path / 'dest' / 'run')
    assert (source / 'a.txt').read_text() == 'new'
    assert (tmp_path / 'dest' / 'run' / 'a.txt').read_text() == 'old'


def test_archive_run_moves_inputs_and_outputs_like_v1(tmp_path):
    """Check archive run moves inputs and outputs like v1."""
    inputs = _tree(tmp_path / 'DMS_uncal', {'x_uncal.fits': 'u'})
    outputs = _tree(tmp_path / 'pipeline_outputs_directory_tag',
                    {'v2/Files/Cost_.txt': 'c'})
    dest = tmp_path / 'archive'
    dest.mkdir()
    messages = []
    cfg = {'archive_to_longterm_storage': str(dest),
           'input_dir': str(inputs) + '/'}
    moved = archive.archive_run(cfg, outputs, logger=messages.append)
    assert moved == {'input': str(dest / 'DMS_uncal'),
                     'output': str(dest / 'pipeline_outputs_directory_tag')}
    assert (dest / 'DMS_uncal' / 'x_uncal.fits').read_text() == 'u'
    assert not inputs.exists() and not outputs.exists()
    # A second call finds nothing to move and only warns (v1 behaviour).
    assert archive.archive_run(cfg, outputs, logger=messages.append) == {}
    assert any('already moved' in message for message in messages)


def test_archive_run_requires_an_existing_destination(tmp_path):
    """Check archive run requires an existing destination."""
    inputs = _tree(tmp_path / 'in', {'a': 'a'})
    messages = []
    moved = archive.archive_run(
        {'archive_to_longterm_storage': str(tmp_path / 'missing'),
         'input_dir': str(inputs)}, tmp_path / 'out',
        logger=messages.append)
    assert moved == {} and inputs.exists()
    assert any('does not exist' in message for message in messages)
    for null in (None, 'None', 'null', ''):
        assert archive.archive_run({'archive_to_longterm_storage': null},
                                   tmp_path, logger=None) == {}
