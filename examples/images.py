"""Ask about a photo, and find a thing in it (TypeSafe, Jev 1.14).

Image support is TypeSafe "Dreamer-ready" (not production-ready): formats, limits
and accuracy may change. The provider adds the `TypeSafe-Danger-Zone-Images` header.

Run:  uv run --env-file .env --extra typesafe python examples/images.py kitchen.jpg
"""

import sys

from typesafe_sdk import TypeSafeClient

from semantic_operators import Boolean, Choice, Image, locate
from semantic_operators.providers.typesafe import TypeSafe

# PNG/JPEG, 256-1568 px per side, upright pixels: resize and rotate before this.
photo = Image.from_path(sys.argv[1])
state = {"photo": photo}

with TypeSafeClient() as client:
    provider = TypeSafe(client, model="jev-1.14.0")  # Box needs 1.14

    answers = provider.ask(state, {
        "indoors": Boolean("Was `photo` taken indoors?"),
        # Options are a name -> description map; descriptions may hold images too.
        "room": Choice("Which room is `photo`?", {
            "kitchen": "Has a stove, sink or fridge.",
            "office": "Has a desk or computer.",
            "other": None,
        }),
    })
    for name, answer in answers.items():
        print(f"{name}: {answer.value} ({answer.confidence:.2f})")

    # Box never says "not found", so locate() asks a presence question with it.
    mug = locate(provider, state, "a coffee mug", where="`photo`")
    if mug.found:
        print("mug at", mug.region.pixels(photo.width, photo.height),
              f"(box confidence {mug.box.confidence:.2f})")
    else:
        print("no mug" if mug.found is False else "not sure there's a mug")
