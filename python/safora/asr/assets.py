"""Offline validation/extraction of existing Vosk Small assets. No runtime network."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

REQUIRED_FILES = (
    'am/final.mdl', 'conf/model.conf', 'conf/mfcc.conf',
    'graph/HCLr.fst', 'graph/Gr.fst', 'graph/disambig_tid.int', 'graph/phones/word_boundary.int',
    'ivector/final.ie', 'ivector/final.dubm', 'ivector/final.mat',
    'ivector/global_cmvn.stats', 'ivector/splice.conf',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_model_dir(path, file_checksums=None):
    """Checks the supported Small layout. Optional per-file hashes are caller-provided.

    An archive SHA is NOT a directory hash. A directory-only path check establishes
    structure, not cryptographic provenance or successful native model parsing.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError('Local Vosk model directory is unavailable')
    for relative in REQUIRED_FILES:
        file = root / relative
        if (not file.is_file() or file.stat().st_size == 0 or file.is_symlink()
                or not file.resolve().is_relative_to(root.resolve())):
            raise ValueError('Incomplete Vosk Small model: ' + relative)
    for relative, expected in (file_checksums or {}).items():
        file = root / relative
        if not file.resolve().is_relative_to(root.resolve()) or file.is_symlink():
            raise ValueError('Unsafe model checksum path')
        if sha256(file) != expected:
            raise ValueError('Vosk model file SHA256 mismatch')
    return root


def validate_archive(path, spec):
    """Validate supplied archive checksum, Small layout and bounded safe ZIP entries."""
    path = Path(path)
    if spec.get('bytes') is not None and path.stat().st_size != spec['bytes']:
        raise ValueError('Vosk archive size mismatch')
    if spec.get('sha256') is not None and sha256(path) != spec['sha256']:
        raise ValueError('Vosk archive SHA256 mismatch')
    if len(PurePosixPath(spec['root']).parts) != 1 or spec['root'] in ('.','..') or '/' in spec['root'] or chr(92) in spec['root']:
        raise ValueError('Unsafe model root')
    prefix = spec['root'] + '/'
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 10000 or sum(info.file_size for info in entries) > 300 * 1024 * 1024:
            raise ValueError('Oversized Vosk archive')
        names = set()
        for info in entries:
            parts = PurePosixPath(info.filename).parts
            if (not info.filename.startswith(prefix) or '..' in parts or chr(92) in info.filename
                    or stat.S_ISLNK(info.external_attr >> 16) or info.filename in names):
                raise ValueError('Unsafe/duplicate Vosk archive path')
            names.add(info.filename)
        for relative in REQUIRED_FILES:
            name = prefix + relative
            if name not in names or archive.getinfo(name).file_size == 0:
                raise ValueError('Incomplete Vosk archive: ' + relative)
        if archive.testzip() is not None:
            raise ValueError('Corrupt Vosk archive')


def prepare_model(archive_path, output, spec):
    """Publish a fresh local directory atomically. Existing directories are never overwritten."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Output already exists; validate or choose a fresh model directory')
    validate_archive(archive_path, spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.vosk-', dir=output.parent) as temp:
        stage = Path(temp) / 'model'
        stage.mkdir()
        prefix = spec['root'] + '/'
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                relative = info.filename.removeprefix(prefix)
                if not relative:
                    continue
                target = stage / relative
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open('wb') as dest:
                        shutil.copyfileobj(source, dest, 65536)
        validate_model_dir(stage)
        hashes = {str(file.relative_to(stage)): sha256(file) for file in sorted(stage.rglob('*')) if file.is_file()}
        (stage / '.safora-model.json').write_text(json.dumps({
            'archive_sha256': sha256(archive_path), 'root': spec['root'],
            'language': spec.get('language'), 'file_sha256': hashes,
        }, indent=2) + '\n')
        stage.rename(output)
    return output
