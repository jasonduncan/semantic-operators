"""Images: pictures as part of what a question is asked about.

An ``Image`` is checked when it's created, so a bad file fails here with a clear
message instead of failing the whole provider call. TypeSafe's current rules
("Dreamer-ready", so they may change):

- PNG or JPEG stills only (no animated PNG), at most 16 MiB per file;
- each side 256-1568 px, total area 78,400-2,458,624 px;
- images are rejected, not resized, and EXIF orientation is not applied, so
  rotate the pixels yourself (an EXIF-rotated JPEG raises a warning here);
- at most 32 MiB of images per request, counting every use of an image.

Put an image anywhere text goes. As a value in ``state`` (``{"photo": image}``),
or inside any text, including instructions and Choice option descriptions, with
``str(image)`` (an f-string works: ``f"Is {logo} on the box in `photo`?"``).
Each image has a unique placeholder, ``str(image)``, that the provider swaps for
the real image. Keep the ``Image`` object alive until the call returns: a
placeholder whose image is gone raises ``ImageError`` rather than being sent
as plain text.

Several frames (from a video, say) go in ``state`` as an array, oldest first.
``frames`` builds that array.
"""

import base64
import binascii
import re
import struct
import uuid
import warnings
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Literal

MIN_SIDE = 256
MAX_SIDE = 1568
MIN_AREA = 78_400
MAX_AREA = 2_458_624
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_REQUEST_BYTES = 32 * 1024 * 1024

Format = Literal["png", "jpeg"]

_PNG = b"\x89PNG\r\n\x1a\n"
_JPEG = b"\xff\xd8\xff"
_PLACEHOLDER = re.compile(r"image-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_live: "weakref.WeakValueDictionary[str, Image]" = weakref.WeakValueDictionary()


class ImageError(ValueError):
    """An image the provider would reject, or a request that breaks the image limits."""


class ImageWarning(UserWarning):
    """An image that will be sent, but probably not as you meant (e.g. EXIF-rotated)."""


@dataclass(frozen=True, eq=False)
class Image:
    """A PNG or JPEG image, checked against the limits above when created."""

    data: bytes = field(repr=False)
    format: Format = field(init=False)
    width: int = field(init=False)
    height: int = field(init=False)
    placeholder: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise ImageError(f"image data must be bytes, got {type(self.data).__name__}")
        data = bytes(self.data)
        fmt, width, height = _inspect(data)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "format", fmt)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "placeholder", f"image-{uuid.uuid4()}")
        _live[self.placeholder] = self

    @classmethod
    def from_path(cls, path: str | PathLike[str]) -> "Image":
        try:
            return cls(Path(path).read_bytes())
        except OSError as error:
            raise ImageError(f"can't read image {str(path)!r}: {error.strerror or error}") from error

    @classmethod
    def from_base64(cls, text: str) -> "Image":
        """From standard base64 (no ``data:`` prefix; whitespace is ignored)."""
        if text.startswith("data:"):
            raise ImageError("base64 image must not have a data: URL prefix")
        try:
            return cls(base64.b64decode("".join(text.split()), validate=True))
        except binascii.Error as error:
            raise ImageError(f"invalid base64 image data: {error}") from error

    def base64(self) -> str:
        """Standard, padded base64, as the ``data_base64`` asset field wants it."""
        return base64.b64encode(self.data).decode("ascii")

    @property
    def size(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return self.placeholder

    def __repr__(self) -> str:
        return f"Image({self.format}, {self.width}x{self.height}, {self.size} bytes)"


def frames(images: Iterable[Image], seconds_ago: Iterable[float] | None = None) -> list[dict[str, Any]]:
    """Still frames as a ``state`` value, oldest first: ``[{"frame": image}, ...]``.

    With ``seconds_ago`` (one per frame) each item also says how long ago it was taken.
    Refer to one frame in instructions as e.g. ```frames[2].frame``` when the array is
    ``state["frames"]``.
    """
    images = list(images)
    if not images:
        raise ImageError("frames needs at least one image")
    if seconds_ago is None:
        return [{"frame": image} for image in images]
    times = list(seconds_ago)
    if len(times) != len(images):
        raise ImageError(f"got {len(images)} frames but {len(times)} seconds_ago values")
    return [{"seconds_ago": t, "frame": image} for t, image in zip(times, images)]


def collect(*values: Any) -> dict[str, tuple[Image, int]]:
    """Every image used in ``values`` (states, questions...), by placeholder, with its use count.

    Finds ``Image`` objects anywhere in dicts, lists, tuples and dataclasses, and
    placeholders inside any text. Raises ``ImageError`` for a placeholder whose image
    no longer exists, or when the images add up to more than ``MAX_REQUEST_BYTES``.
    """
    found: dict[str, tuple[Image, int]] = {}

    def use(image: Image) -> None:
        _, count = found.get(image.placeholder, (image, 0))
        found[image.placeholder] = (image, count + 1)

    def walk(value: Any) -> None:
        if isinstance(value, Image):
            use(value)
        elif isinstance(value, str):
            for placeholder in _PLACEHOLDER.findall(value):
                image = _live.get(placeholder)
                if image is None:
                    raise ImageError(f"{placeholder} refers to an Image that no longer exists; "
                                     "keep the Image object alive until the call returns")
                use(image)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                walk(k)
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)
        elif hasattr(value, "__dataclass_fields__"):
            for name in value.__dataclass_fields__:
                walk(getattr(value, name))

    for value in values:
        walk(value)
    unique = sum(image.size for image, _ in found.values())
    expanded = sum(image.size * count for image, count in found.values())
    if unique > MAX_REQUEST_BYTES or expanded > MAX_REQUEST_BYTES:
        raise ImageError(f"images in one request must total at most 32 MiB, counting each use "
                         f"(got {expanded / 2**20:.1f} MiB); send fewer or smaller images")
    return found


def replace_images(value: Any) -> Any:
    """``value`` with every ``Image`` object replaced by its placeholder text."""
    if isinstance(value, Image):
        return value.placeholder
    if isinstance(value, Mapping):
        return {k: replace_images(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [replace_images(v) for v in value]
    return value


# Checking the file


def _inspect(data: bytes) -> tuple[Format, int, int]:
    if len(data) > MAX_FILE_BYTES:
        raise ImageError(f"image is {len(data) / 2**20:.1f} MiB; the limit is 16 MiB per file")
    if data.startswith(_PNG):
        fmt: Format = "png"
        width, height = _png(data)
    elif data.startswith(_JPEG):
        fmt = "jpeg"
        width, height = _jpeg(data)
    else:
        raise ImageError("image must be PNG or JPEG (unrecognized file signature)")
    for side, n in (("width", width), ("height", height)):
        if not MIN_SIDE <= n <= MAX_SIDE:
            raise ImageError(f"image {side} is {n} px; each side must be {MIN_SIDE}-{MAX_SIDE} px "
                             f"(images are rejected, not resized: resize it first)")
    if not MIN_AREA <= width * height <= MAX_AREA:
        raise ImageError(f"image area is {width * height} px; it must be {MIN_AREA}-{MAX_AREA} px")
    return fmt, width, height


def _png(data: bytes) -> tuple[int, int]:
    if len(data) < 33 or data[12:16] != b"IHDR":
        raise ImageError("malformed PNG: no IHDR header")
    width, height = struct.unpack(">II", data[16:24])
    pos = 8
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        if kind == b"acTL":
            raise ImageError("animated PNGs are not accepted; send a still frame")
        if kind in (b"IDAT", b"IEND"):
            break
        pos += 12 + length
    return width, height


_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _jpeg(data: bytes) -> tuple[int, int]:
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            raise ImageError("malformed JPEG: bad marker")
        marker = data[pos + 1]
        if marker == 0xFF:
            pos += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        (length,) = struct.unpack(">H", data[pos + 2:pos + 4])
        segment = data[pos + 4:pos + 2 + length]
        if marker == 0xE1 and segment.startswith(b"Exif\x00\x00"):
            orientation = _exif_orientation(segment[6:])
            if orientation not in (None, 1):
                warnings.warn(f"JPEG has EXIF orientation {orientation}, which TypeSafe ignores: "
                              "the model sees the pixels unrotated. Rotate them yourself.",
                              ImageWarning, stacklevel=4)
        if marker in _SOF:
            if len(segment) < 5:
                break
            height, width = struct.unpack(">HH", segment[1:5])
            if marker != 0xC0 and marker != 0xC1:
                warnings.warn("JPEG is not baseline; TypeSafe recommends baseline JPEG",
                              ImageWarning, stacklevel=4)
            return width, height
        if marker == 0xDA:
            break
        pos += 2 + length
    raise ImageError("malformed JPEG: no frame header with its size")


def _exif_orientation(tiff: bytes) -> int | None:
    try:
        endian = {b"II": "<", b"MM": ">"}[tiff[:2]]
        (ifd,) = struct.unpack(endian + "I", tiff[4:8])
        (count,) = struct.unpack(endian + "H", tiff[ifd:ifd + 2])
        for i in range(count):
            entry = tiff[ifd + 2 + 12 * i: ifd + 14 + 12 * i]
            tag, _, _ = struct.unpack(endian + "HHI", entry[:8])
            if tag == 0x0112:
                return struct.unpack(endian + "H", entry[8:10])[0]
    except (KeyError, struct.error):
        return None
    return None
