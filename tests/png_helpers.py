"""PNGs mínimos RGB para testar cache sem Blender ou dependências de imagem."""

import json
import struct
import zlib
from pathlib import Path

from src.frame_cache import render_dimensions, valid_png


def png_bytes(width: int, height: int, value: int = 0) -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = (b"\0" + bytes([value]) * (width * 3)) * height
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def finish_frames(command, value: int = 0) -> None:
    plan = json.loads(Path(command[command.index("--scene-json") + 1]).read_text())
    frames = Path(command[command.index("--frames-dir") + 1])
    width, height = render_dimensions(plan["resolution_percentage"])
    image = png_bytes(width, height, value)
    for frame in range(1, plan["frame_end"] + 1):
        path = frames / f"frame_{frame:06d}.png"
        if not valid_png(path, width, height):
            path.write_bytes(image)
