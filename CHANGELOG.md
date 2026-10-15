# Changelog

## 0.12.0 (unreleased)

Image support for TypeSafe (Jev 1.13+; Box needs `jev-1.14.0`). TypeSafe labels images
"Dreamer-ready" (not production-ready): formats, limits and accuracy may change.

- `Image` (`Image(bytes)`, `Image.from_path`, `Image.from_base64`): PNG/JPEG stills, checked
  when created. Animated PNGs are rejected. Each image must be at most 16 MiB, with each side
  256-1568 px and an area of 78,400-2,458,624 px. EXIF-rotated and non-baseline JPEGs raise
  `ImageWarning`; bad images raise `ImageError` (a `ValueError`).
- Images go anywhere text goes: as state values, or as `str(image)` inside instructions and
  option/level descriptions. The TypeSafe provider sends them as multipart assets with
  `TypeSafe-Danger-Zone-Images: true`, and checks the 32 MiB per-request limit (counting
  each use) before sending.
- `Box` questions return a `Region` (`top`/`bottom`/`left`/`right` as fractions, plus
  `confidence`), with `padded()`, `pixels()` and `center`.
- `locate()` / `locate_async()`: a presence `Boolean` and a `Box` in one call, because Box
  has no "not found" answer.
- `frames()` builds a multi-frame state (an array of frames, oldest first).
- Laya refuses images and Box questions; benchmarks don't score Box answers.
- Requires `typesafe-sdk>=0.7.5` for the `typesafe` extra.

Earlier releases are described in the GitHub release notes.
