"""Images and Box: client-side checks and the TypeSafe request shape. Offline (mocked HTTP)."""

import base64
import gc
import json

import httpx2
import pytest
import typesafe_sdk as ts

from imagekit import jpeg_header, png
from semantic_operators import (Boolean, Box, Choice, Image, ImageError, ImageWarning, Region,
                                frames, locate, make_answer)
from semantic_operators.images import MAX_REQUEST_BYTES, collect
from semantic_operators.providers.typesafe import TypeSafe


# Client-side checks

def test_png_and_jpeg_sizes_are_read():
    image = Image(png(300, 400))
    assert (image.format, image.width, image.height) == ("png", 300, 400)
    image = Image(jpeg_header(1568, 1000))
    assert (image.format, image.width, image.height) == ("jpeg", 1568, 1000)


@pytest.mark.parametrize("data, message", [
    (b"GIF89a" + b"\x00" * 100, "PNG or JPEG"),
    (png(255, 400), "width is 255 px"),
    (png(400, 1569), "height is 1569 px"),
    (png(animated=True), "animated"),
    (jpeg_header(100, 300), "width is 100 px"),
    (b"\xff\xd8\xff\xd9", "malformed JPEG"),
])
def test_bad_images_are_rejected_with_a_clear_message(data, message):
    with pytest.raises(ImageError, match=message):
        Image(data)


def test_area_limit():
    Image(jpeg_header(1568, 1568))  # exactly the max area
    Image(png(280, 280))             # exactly the min area
    with pytest.raises(ImageError, match="area"):
        Image(png(256, 256))         # both sides allowed, area too small


def test_file_size_limit():
    with pytest.raises(ImageError, match="16 MiB"):
        Image(png() + b"\x00" * (16 * 1024 * 1024))


def test_exif_rotation_and_progressive_jpeg_warn():
    with pytest.warns(ImageWarning, match="orientation 6"):
        Image(jpeg_header(orientation=6))
    with pytest.warns(ImageWarning, match="baseline"):
        Image(jpeg_header(sof=0xC2))


def test_base64_round_trip_and_data_url_rejected():
    image = Image(png())
    assert Image.from_base64(image.base64()).data == image.data
    with pytest.raises(ImageError, match="data: URL"):
        Image.from_base64("data:image/png;base64," + image.base64())
    with pytest.raises(ImageError, match="base64"):
        Image.from_base64("not base64!")


def test_from_path(tmp_path):
    path = tmp_path / "x.png"
    path.write_bytes(png())
    assert Image.from_path(path).format == "png"
    with pytest.raises(ImageError, match="can't read"):
        Image.from_path(tmp_path / "missing.png")


def test_collect_counts_each_use_and_enforces_request_limit():
    a, b = Image(png()), Image(png(300, 300))
    found = collect({"photo": a, "text": f"compare {a} with {b}"}, Boolean(f"Is {b} red?"))
    assert {k: n for k, (_, n) in found.items()} == {a.placeholder: 2, b.placeholder: 2}

    big = Image(png())
    object.__setattr__(big, "data", b"\x00" * (MAX_REQUEST_BYTES // 2 + 1))
    with pytest.raises(ImageError, match="32 MiB"):
        collect([big, big])  # one image, used twice: counted twice


def test_placeholder_of_a_vanished_image_is_an_error():
    text = f"Is {Image(png())} red?"
    gc.collect()
    with pytest.raises(ImageError, match="no longer exists"):
        collect(text)


def test_frames():
    a, b = Image(png()), Image(png())
    assert frames([a, b]) == [{"frame": a}, {"frame": b}]
    assert frames([a, b], [2, 0]) == [{"seconds_ago": 2, "frame": a}, {"seconds_ago": 0, "frame": b}]
    with pytest.raises(ImageError):
        frames([a], [1, 2])


# Box answers

def test_box_answer_and_region_helpers():
    q = Box("the mug", min_confidence=0.5)
    answer = make_answer(q, Region(0.2, 0.6, 0.1, 0.5), {}, confidence=0.9)
    assert answer.decided and answer.confidence == 0.9
    assert make_answer(q, Region(0.2, 0.6, 0.1, 0.5), {}, confidence=0.4).value is None
    assert make_answer(Box("x"), Region(0, 1, 0, 1), {}).confidence == 0.0  # none reported
    r = Region(0.2, 0.6, 0.1, 0.5)
    assert r.pixels(1000, 500) == (100, 100, 500, 300)
    assert r.padded(1.0) == Region(0.0, 1.0, 0.0, 0.9)
    with pytest.raises(ValueError):
        Region(0.6, 0.2, 0, 1)
    with pytest.raises(ValueError):
        make_answer(q, "not a region", {}, confidence=0.9)
    with pytest.raises(ValueError):
        make_answer(Boolean("x"), True, {"true": 1, "false": 0}, confidence=1)


# The TypeSafe request

def _client(answers, seen):
    def handle(request):
        seen.append(request)
        return httpx2.Response(200, json={"model": "jev-1.14.0",
                                          "usage": {"input_tokens": 1, "output_tokens": 0},
                                          "answers": answers})
    return ts.TypeSafeClient(api_key="test", transport=httpx2.MockTransport(handle),
                             retry=ts.RetryPolicy(max_retries=0))


def _multipart(request):
    body = request.read()
    ctype = request.headers["content-type"]
    assert ctype.startswith("multipart/form-data")
    boundary = ctype.split("boundary=")[1].encode()
    parts = {}
    for part in body.split(b"--" + boundary)[1:-1]:
        head, _, content = part.partition(b"\r\n\r\n")
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        parts[name] = content[:-2]
    return parts


BOX = {"type": "box", "box": {"top": 0.25, "bottom": 0.82, "left": 0.44, "right": 0.84},
       "confidence": 0.92}


def test_images_are_sent_as_multipart_assets_with_the_header():
    photo, logo = Image(png()), Image(png(300, 300))
    seen = []
    provider = TypeSafe(_client({"mug": BOX, "brand": {"type": "choice", "choice": "a",
                                 "confidence": 0.8, "probabilities": {"a": 0.8, "b": 0.2}}}, seen),
                        model="jev-1.14.0")
    answers = provider.ask({"photo": photo}, {
        "mug": Box("the coffee mug in `photo`"),
        "brand": Choice("Which brand is on the mug?", {"a": f"this logo: {logo}", "b": "other"}),
    })
    assert answers["mug"].value == Region(0.25, 0.82, 0.44, 0.84)
    assert answers["mug"].confidence == 0.92

    request = seen[0]
    assert request.headers["TypeSafe-Danger-Zone-Images"] == "true"
    parts = _multipart(request)
    body = json.loads(parts["request"])
    assert body["state"] == {"photo": photo.placeholder}
    assert body["questions"]["mug"] == {"type": "box", "instructions": "the coffee mug in `photo`"}
    assert body["questions"]["brand"]["criteria"]["a"] == f"this logo: {logo.placeholder}"
    assert set(body["assets"]) == {photo.placeholder, logo.placeholder}
    for image in (photo, logo):
        part = body["assets"][image.placeholder]["part"]
        assert parts[part] == image.data


def test_text_only_requests_are_unchanged():
    seen = []
    TypeSafe(_client({"q": {"type": "noul", "noul": 0.9}}, seen)).ask("hi", {"q": Boolean("Is it?")})
    assert "TypeSafe-Danger-Zone-Images" not in seen[0].headers
    assert "assets" not in json.loads(seen[0].read())


def test_a_bad_image_fails_before_sending():
    seen = []
    a = Image(png())
    object.__setattr__(a, "data", b"\x00" * (MAX_REQUEST_BYTES + 1))
    with pytest.raises(ImageError):
        TypeSafe(_client({}, seen)).ask(a, {"q": Boolean("Is it?")})
    assert seen == []


def test_multi_frame_state():
    a, b = Image(png()), Image(png())
    seen = []
    TypeSafe(_client({"q": {"type": "noul", "noul": 0.2}}, seen)).ask(
        {"frames": frames([a, b], [2, 0])}, {"q": Boolean("Did the door open during `frames`?")})
    body = json.loads(_multipart(seen[0])["request"])
    assert body["state"]["frames"] == [{"seconds_ago": 2, "frame": a.placeholder},
                                       {"seconds_ago": 0, "frame": b.placeholder}]


@pytest.mark.parametrize("p_present, found, has_region", [(0.9, True, True), (0.1, False, False),
                                                          (0.5, None, False)])
def test_locate_pairs_box_with_a_presence_check(p_present, found, has_region):
    seen = []
    provider = TypeSafe(_client({"present": {"type": "noul", "noul": p_present}, "box": BOX}, seen),
                        model="jev-1.14.0")
    result = locate(provider, {"photo": Image(png())}, "a coffee mug", where="`photo`")
    assert result.found is found
    assert (result.region is not None) == has_region
    body = json.loads(_multipart(seen[0])["request"])
    assert body["questions"]["present"]["instructions"] == "Is there a coffee mug in `photo`?"


def test_laya_refuses_images_and_box():
    from semantic_operators import ProviderError
    from semantic_operators.providers.laya import Laya

    class Model:
        def predict(self, state, qs):
            raise AssertionError("must not be called")

    with pytest.raises(ProviderError, match="not supported"):
        Laya(Model()).ask({"photo": Image(png())}, {"q": Boolean("Is it?")})
    with pytest.raises(ProviderError, match="not supported"):
        Laya(Model()).ask("text", {"b": Box("the mug")})
