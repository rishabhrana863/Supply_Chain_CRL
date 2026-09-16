"""Open the exact frozen ZIP, reconstructing checked parts when necessary."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile


@contextmanager
def open_frozen_archive(path):
    path = Path(path)
    descriptor = path.with_suffix(path.suffix + '.parts.json')
    spec = json.loads(descriptor.read_text()) if descriptor.exists() else None
    if path.exists():
        if spec is not None:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != spec['sha256']:
                raise ValueError('Frozen archive checksum mismatch')
        with path.open('rb') as stream:
            yield stream
        return
    if spec is None:
        raise FileNotFoundError(f'Neither frozen archive nor part descriptor exists: {path}')
    whole = hashlib.sha256()
    total = 0
    with tempfile.TemporaryFile() as stream:
        for part in spec['parts']:
            relative = Path(part['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe archive-part path')
            data = (path.parent / relative).read_bytes()
            if len(data) != part['bytes'] or hashlib.sha256(data).hexdigest() != part['sha256']:
                raise ValueError(f'Frozen archive part mismatch: {relative}')
            stream.write(data)
            whole.update(data)
            total += len(data)
        if total != spec['bytes'] or whole.hexdigest() != spec['sha256']:
            raise ValueError('Reconstructed frozen archive mismatch')
        stream.seek(0)
        yield stream
