"""Tiny PNG/JPEG builders for tests (no Pillow needed)."""

import struct
import zlib


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png(width: int = 280, height: int = 280, animated: bool = False) -> bytes:
    """A real, decodable grayscale PNG."""
    raw = b"".join(b"\x00" + b"\x80" * width for _ in range(height))
    body = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    if animated:
        body += _chunk(b"acTL", struct.pack(">II", 2, 0))
    return b"\x89PNG\r\n\x1a\n" + body + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b"")


def jpeg_header(width: int = 300, height: int = 300, *, sof: int = 0xC0,
                orientation: int | None = None) -> bytes:
    """Just enough JPEG structure to read the size (not decodable)."""
    out = b"\xff\xd8"
    if orientation is not None:
        tiff = b"MM\x00\x2a\x00\x00\x00\x08" + struct.pack(">H", 1) \
            + struct.pack(">HHIHH", 0x0112, 3, 1, orientation, 0) + b"\x00\x00\x00\x00"
        seg = b"Exif\x00\x00" + tiff
        out += b"\xff\xe1" + struct.pack(">H", len(seg) + 2) + seg
    sof_body = struct.pack(">BHHB", 8, height, width, 1) + b"\x01\x11\x00"
    out += bytes([0xFF, sof]) + struct.pack(">H", len(sof_body) + 2) + sof_body
    return out + b"\xff\xd9"
