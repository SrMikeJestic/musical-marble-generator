"""Cache persistente de frames PNG; usa apenas a biblioteca padrão do Python."""

import hashlib
import json
import os
import shutil
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from .config import RESOLUTION_X


def render_dimensions(percentage: int) -> tuple[int, int]:
    """Dimensões pares em 9:16, iguais às usadas pelo builder."""
    unit = max(2, 2 * round((RESOLUTION_X / 9) * percentage / 200))
    return 9 * unit, 16 * unit


def valid_png(path: Path, width: int, height: int) -> bool:
    """Verifica PNG RGB 8-bit, dimensões, CRCs e dados zlib até o IEND.

    A leitura é incremental e a descompressão fica limitada ao tamanho esperado
    do frame. Arquivos truncados após desconexões nunca contam como progresso.
    """
    try:
        with path.open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                return False
            decoder = zlib.decompressobj()
            expected_bytes = height * (width * 3 + 1)
            decoded_bytes = 0
            header_seen = data_seen = False
            while True:
                header = stream.read(8)
                if len(header) != 8:
                    return False
                length, kind = struct.unpack(">I4s", header)
                if not header_seen and (kind != b"IHDR" or length != 13):
                    return False
                if kind == b"IHDR" and (header_seen or length != 13):
                    return False
                crc = zlib.crc32(kind)
                remaining = length
                chunk_header = b""
                while remaining:
                    data = stream.read(min(remaining, 65536))
                    if not data:
                        return False
                    remaining -= len(data)
                    crc = zlib.crc32(data, crc)
                    if kind == b"IHDR":
                        chunk_header += data
                    elif kind == b"IDAT":
                        if not header_seen:
                            return False
                        data_seen = True
                        decoded_bytes += len(decoder.decompress(data, expected_bytes - decoded_bytes + 1))
                        if decoded_bytes > expected_bytes or decoder.unconsumed_tail or decoder.unused_data:
                            return False
                checksum = stream.read(4)
                if len(checksum) != 4 or struct.unpack(">I", checksum)[0] != (crc & 0xFFFFFFFF):
                    return False
                if kind == b"IHDR":
                    if header_seen or struct.unpack(">IIBBBBB", chunk_header) != (width, height, 8, 2, 0, 0, 0):
                        return False
                    header_seen = True
                elif kind == b"IEND":
                    return (length == 0 and data_seen and decoder.eof and
                            decoded_bytes == expected_bytes and stream.read(1) == b"")
    except (OSError, ValueError, struct.error, zlib.error):
        return False


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_identity(midi_path: Path, plan: dict, source_root: Path) -> dict:
    """Inclui câmera/materiais/configuração no hash, além dos bytes do MIDI."""
    payload = {
        "cache_version": 1,
        "midi_sha256": file_digest(midi_path),
        "plan": plan,
        "sources": {name: file_digest(source_root / name) for name in (
            "blender_builder.py", "config.py", "layout_solver.py", "frame_cache.py",
        )},
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"key": hashlib.sha256(serialized.encode()).hexdigest(), **payload}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@dataclass
class FrameStatus:
    valid: set[int]
    invalid: list[Path]
    expected: int

    @property
    def complete(self) -> bool:
        return len(self.valid) == self.expected and not self.invalid


class FrameCache:
    def __init__(self, directory: Path, frame_count: int, width: int, height: int):
        self.directory = Path(directory)
        self.frames = self.directory / "frames"
        self.frame_count = frame_count
        self.width = width
        self.height = height

    def frame_path(self, frame: int) -> Path:
        return self.frames / f"frame_{frame:06d}.png"

    def prepare(self, identity: dict, *, force: bool = False) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = self.directory / "cache.json"
        try:
            previous = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
        if force or previous != identity:
            # Só remove a sequência deste job; nunca apaga a raiz de --cache-dir.
            if self.frames.is_symlink():
                raise RuntimeError("O diretório de frames do cache não pode ser um link simbólico.")
            if self.frames.exists():
                shutil.rmtree(self.frames)
        self.frames.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest, identity)

    def inspect(self) -> FrameStatus:
        names = {self.frame_path(frame).name: frame for frame in range(1, self.frame_count + 1)}
        valid = set()
        invalid = []
        if self.frames.exists():
            for path in self.frames.iterdir():
                if path.suffix.lower() != ".png":
                    continue
                frame = names.get(path.name)
                if frame is not None and valid_png(path, self.width, self.height):
                    valid.add(frame)
                else:
                    invalid.append(path)
        return FrameStatus(valid, invalid, self.frame_count)

    def prune_invalid(self, status: FrameStatus) -> None:
        for path in status.invalid:
            path.unlink()
