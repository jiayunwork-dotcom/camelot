"""Reconcile PDF text coordinates with the rasterised (visible) page.

Camelot feeds a single-page PDF to two independent consumers:

* PDFMiner, for the text layout.  PDFMiner ignores the page ``CropBox`` and
  lays the page out in the rotated ``MediaBox`` frame (lower-left origin,
  rotation applied, MediaBox origin removed).
* PDFium (or another image backend), for the line detection image.  The
  renderer only rasterises the visible page: the ``CropBox`` clipped to the
  ``MediaBox``, with the page rotation applied.

For uncropped pages whose ``MediaBox`` starts at ``(0, 0)`` both frames are
identical, which is why the historical ``pdf_width``/``image_width`` scaling
worked.  As soon as the page is cropped (or the MediaBox origin is offset, or
the page is rotated) the two frames differ and every coordinate has to go
through an explicit mapping.

This module is the single place that owns that mapping.  Page geometry is
resolved once, in the page-preparation layer (:mod:`camelot.handlers`), and the
resulting :class:`PageGeometry` is handed to the parsers.  Parsers never guess
the page geometry from image dimensions again.

All public coordinates stay in the *uncropped* page frame: the frame produced
by the same page with the ``CropBox`` removed while keeping its rotation and
``MediaBox``.  Cropping therefore only hides content; it never moves the
coordinates returned to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Iterable
from typing import Sequence


Box = tuple[float, float, float, float]


def normalize_box(box: Sequence[float]) -> Box:
    """Return ``box`` as ``(x0, y0, x1, y1)`` with ``x0 <= x1``, ``y0 <= y1``."""
    x0, y0, x1, y1 = (float(v) for v in box)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def intersect_boxes(a: Box, b: Box) -> Box | None:
    """Return the intersection of two boxes, or ``None`` if they don't overlap."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def rotation_of(page: Any) -> int:
    """Return the normalised clockwise page rotation of a pypdf page."""
    try:
        rotation = int(page.rotation)
    except Exception:  # pragma: no cover - defensive, pypdf always exposes it
        rotation = 0
    rotation %= 360
    # PDFium only supports multiples of 90; anything else is displayed as-is.
    if rotation % 90:
        return 0
    return rotation


def map_page_point(x: float, y: float, media_box: Box, rotation: int) -> tuple[float, float]:
    """Map a PDF user-space point into the PDFMiner layout frame.

    This mirrors the CTM PDFMiner applies in ``PDFPageInterpreter``: the
    MediaBox origin is removed and the page rotation is applied so that the
    layout is upright on screen.
    """
    mx0, my0, mx1, my1 = media_box
    rotation %= 360
    if rotation == 90:
        return (y - my0, mx1 - x)
    if rotation == 180:
        return (mx1 - x, my1 - y)
    if rotation == 270:
        return (my1 - y, x - mx0)
    return (x - mx0, y - my0)


def visible_bbox(media_box: Box, crop_box: Box, rotation: int) -> Box:
    """Return the visible area in the PDFMiner layout frame."""
    visible = intersect_boxes(normalize_box(media_box), normalize_box(crop_box))
    if visible is None:
        # A CropBox entirely outside the MediaBox shows nothing.  Keep a
        # non-degenerate frame so downstream scaling stays well defined.
        visible = normalize_box(media_box)
    x0, y0, x1, y1 = visible
    corners = [
        map_page_point(x0, y0, media_box, rotation),
        map_page_point(x1, y0, media_box, rotation),
        map_page_point(x0, y1, media_box, rotation),
        map_page_point(x1, y1, media_box, rotation),
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return (min(xs), min(ys), max(xs), max(ys))


@dataclass(frozen=True)
class PageGeometry:
    """Geometry of one page, shared by the text and raster layers.

    Attributes
    ----------
    media_box : tuple
        Normalised ``MediaBox`` in PDF user space.
    crop_box : tuple
        Normalised ``CropBox`` in PDF user space (falls back to the MediaBox).
    rotation : int
        Clockwise page rotation (0, 90, 180 or 270).
    visible_bbox : tuple
        Visible ``(x0, y0, x1, y1)`` in the PDFMiner layout frame, i.e. the
        frame used by the text parser and by the public table coordinates.
    """

    media_box: Box
    crop_box: Box
    rotation: int
    visible_bbox: Box

    @classmethod
    def from_page(cls, page: Any) -> "PageGeometry":
        """Build the geometry from a pypdf ``PageObject``."""
        media = normalize_box(page.mediabox)
        try:
            crop = normalize_box(page.cropbox)
        except Exception:  # pragma: no cover - pypdf falls back to MediaBox
            crop = media
        rotation = rotation_of(page)
        return cls(
            media_box=media,
            crop_box=crop,
            rotation=rotation,
            visible_bbox=visible_bbox(media, crop, rotation),
        )

    @classmethod
    def full_page(cls, width: float, height: float) -> "PageGeometry":
        """Geometry of an implicit, uncropped page (legacy fallback)."""
        box = (0.0, 0.0, float(width), float(height))
        return cls(media_box=box, crop_box=box, rotation=0, visible_bbox=box)

    @property
    def is_cropped(self) -> bool:
        """Whether the visible area differs from the uncropped page."""
        media = self.media_box
        crop = intersect_boxes(media, self.crop_box) or media
        return any(
            abs(a - b) > 1e-6 for a, b in zip(media, crop)  # noqa: B905
        )

    def scale_factors(
        self, image_size: tuple[int, int]
    ) -> tuple[float, float]:
        """Return ``(sx, sy)`` mapping layout coordinates to image pixels."""
        vx0, vy0, vx1, vy1 = self.visible_bbox
        width = vx1 - vx0
        height = vy1 - vy0
        if width <= 0 or height <= 0:
            raise ValueError("Degenerate visible page area")
        return (image_size[0] / width, image_size[1] / height)


def pdf_to_image_point(
    x: float, y: float, geometry: PageGeometry, image_size: tuple[int, int]
) -> tuple[float, float]:
    """Map a layout-frame point to image pixel coordinates (top-left origin)."""
    vx0, _vy0, _vx1, vy1 = geometry.visible_bbox
    sx, sy = geometry.scale_factors(image_size)
    return ((x - vx0) * sx, (vy1 - y) * sy)


def image_to_pdf_point(
    x: float, y: float, geometry: PageGeometry, image_size: tuple[int, int]
) -> tuple[float, float]:
    """Map an image pixel coordinate back to the layout frame."""
    vx0, _vy0, _vx1, vy1 = geometry.visible_bbox
    sx, sy = geometry.scale_factors(image_size)
    return (vx0 + x / sx, vy1 - y / sy)


def pdf_to_image_area(
    area: Sequence[float],
    geometry: PageGeometry,
    image_size: tuple[int, int],
    clip: bool = True,
) -> tuple[int, int, int, int] | None:
    """Map a layout-frame area to ``(x, y, width, height)`` in image pixels.

    The returned rectangle uses the top-left origin expected by
    :func:`camelot.image_processing.find_joints` and
    :func:`camelot.image_processing.find_lines`.  When ``clip`` is true the
    rectangle is clamped to the image (so it is safe to use as a numpy slice)
    and ``None`` is returned when the area does not intersect the visible page.
    When ``clip`` is false the rectangle may extend outside the image, which is
    useful to preserve a caller-provided area as a table bounding box.
    """
    x1, y1, x2, y2 = (float(v) for v in area)
    # ``y1`` is the lower coordinate, so it maps to the larger pixel row.
    left, top = pdf_to_image_point(min(x1, x2), max(y1, y2), geometry, image_size)
    right, bottom = pdf_to_image_point(max(x1, x2), min(y1, y2), geometry, image_size)
    if clip:
        left = max(left, 0.0)
        top = max(top, 0.0)
        right = min(right, float(image_size[0]))
        bottom = min(bottom, float(image_size[1]))
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return None
    return (
        int(round(left)),
        int(round(top)),
        int(round(width)),
        int(round(height)),
    )


def image_to_pdf_bbox(
    bbox: Sequence[float], geometry: PageGeometry, image_size: tuple[int, int]
) -> Box:
    """Map an image-frame ``(x0, y0, x1, y1)`` bounding box to the layout frame."""
    p0 = image_to_pdf_point(float(bbox[0]), float(bbox[1]), geometry, image_size)
    p1 = image_to_pdf_point(float(bbox[2]), float(bbox[3]), geometry, image_size)
    return (
        min(p0[0], p1[0]),
        min(p0[1], p1[1]),
        max(p0[0], p1[0]),
        max(p0[1], p1[1]),
    )


def image_to_pdf_segment(
    segment: Sequence[float], geometry: PageGeometry, image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Map a ``(x1, y1, x2, y2)`` image-frame segment to the layout frame."""
    p1 = image_to_pdf_point(float(segment[0]), float(segment[1]), geometry, image_size)
    p2 = image_to_pdf_point(float(segment[2]), float(segment[3]), geometry, image_size)
    return (p1[0], p1[1], p2[0], p2[1])


def image_to_pdf_tables(
    tables: dict[tuple[float, float, float, float], list[tuple[float, float]]],
    v_segments: Iterable[Sequence[float]],
    h_segments: Iterable[Sequence[float]],
    geometry: PageGeometry,
    image_size: tuple[int, int],
) -> tuple[
    dict[tuple[float, float, float, float], dict[str, list[tuple[float, float]]]],
    list[tuple[float, float, float, float]],
    list[tuple[float, float, float, float]],
]:
    """Translate a whole line-detection result back to the layout frame.

    ``tables`` maps image-frame table bounding boxes (left, bottom, right, top)
    to their joints.  Segments are ``(x1, y1, x2, y2)`` image-frame tuples.
    """
    tables_new: dict[
        tuple[float, float, float, float], dict[str, list[tuple[float, float]]]
    ] = {}
    for bbox, joints in tables.items():
        pdf_bbox = image_to_pdf_bbox(bbox, geometry, image_size)
        joints_new = [
            image_to_pdf_point(joint[0], joint[1], geometry, image_size)
            for joint in joints
        ]
        tables_new[pdf_bbox] = {"joints": joints_new}

    v_new = [
        image_to_pdf_segment(segment, geometry, image_size)
        for segment in v_segments
    ]
    h_new = [
        image_to_pdf_segment(segment, geometry, image_size)
        for segment in h_segments
    ]
    return tables_new, v_new, h_new


def _object_bbox(obj: Any) -> Box | None:
    try:
        return normalize_box((obj.x0, obj.y0, obj.x1, obj.y1))
    except Exception:  # pragma: no cover - defensive
        return None


def filter_objects_in_bbox(objects: Iterable[Any], bbox: Box, tolerance: float = 2.0) -> list[Any]:
    """Keep only objects that belong to ``bbox``.

    An object belongs to the area when the centre of its bounding box lies
    inside it (plus ``tolerance``).  This is the same criterion Camelot already
    uses in :func:`camelot.utils.text_in_bbox` to decide whether a text line is
    part of a table area, so filtering text to the visible page is consistent
    with how it would later be assigned to cells.

    Objects whose centre is off-page are discarded even when they overlap the
    border; only the visible part of such an object would be on the page and
    using its full extent would shift table detection.
    """
    x0, y0, x1, y1 = bbox
    kept = []
    for obj in objects:
        obj_bbox = _object_bbox(obj)
        if obj_bbox is None:
            kept.append(obj)
            continue
        cx = (obj_bbox[0] + obj_bbox[2]) / 2.0
        cy = (obj_bbox[1] + obj_bbox[3]) / 2.0
        if (
            x0 - tolerance <= cx <= x1 + tolerance
            and y0 - tolerance <= cy <= y1 + tolerance
        ):
            kept.append(obj)
    return kept
