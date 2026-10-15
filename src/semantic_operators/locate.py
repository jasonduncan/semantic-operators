"""Find one thing in an image: a presence check plus a Box, in one provider call.

A Box question always returns a box, even when the thing isn't there, and its
confidence isn't a presence signal. ``locate`` asks a yes/no presence question
alongside it and only reports the region when the answer is a confident "yes".
"""

from dataclasses import dataclass

from .provider import AsyncProvider, Provider
from .types import Answer, Boolean, Box, Region, State


@dataclass(frozen=True)
class Located:
    """``found`` is True/False, or None when the presence answer was undecided.

    ``region`` is set only when ``found`` is True. ``presence`` and ``box`` are the
    two raw answers (the box answer is there even when nothing was found).
    """

    found: bool | None
    region: Region | None
    presence: Answer
    box: Answer


def _questions(target: str, where: str, min_presence: float, min_box_confidence: float):
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a non-empty description")
    return {
        "present": Boolean(f"Is there {target} in {where}?", min_confidence=min_presence),
        "box": Box(f"{target} in {where}", min_confidence=min_box_confidence),
    }


def _located(answers: dict[str, Answer]) -> Located:
    presence, box = answers["present"], answers["box"]
    found = presence.value
    region = box.value if found is True and isinstance(box.value, Region) else None
    return Located(found if isinstance(found, bool) else None, region, presence, box)


def locate(provider: Provider, state: State, target: str, *, where: str = "the image",
           min_presence: float = 0.5, min_box_confidence: float = 0.0) -> Located:
    """Is ``target`` (e.g. "a coffee mug") in the image, and where?

    ``where`` names the image the way the questions should, e.g. "`photo`" when the
    state is ``{"photo": image}``. ``min_presence`` is the confidence the yes/no answer
    needs; below it ``found`` is None. With ``min_box_confidence`` a low-confidence box
    gives ``found=True`` but ``region=None``. Use a model that supports Box
    (TypeSafe: ``jev-1.14.0``).
    """
    return _located(provider.ask(state, _questions(target, where, min_presence, min_box_confidence)))


async def locate_async(provider: AsyncProvider, state: State, target: str, *,
                       where: str = "the image", min_presence: float = 0.5,
                       min_box_confidence: float = 0.0) -> Located:
    """``locate`` for an ``AsyncProvider``."""
    questions = _questions(target, where, min_presence, min_box_confidence)
    return _located(await provider.ask(state, questions))
