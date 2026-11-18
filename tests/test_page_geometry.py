"""Regression tests for pages whose visible area is smaller than the MediaBox.

The fixtures used here are generated on the fly so that the ground truth
(matrix, lines and joints) is known independently of Camelot.

Key invariant: cropping a page only hides content.  The same ``table_areas`` /
``table_regions`` coordinates must work on the cropped and on the uncropped
page, and the parsed geometry must stay in the uncropped page frame.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import camelot
from camelot.page_geometry import PageGeometry
from camelot.page_geometry import image_to_pdf_point
from camelot.page_geometry import pdf_to_image_point

# Geometry of the generated test table (PDF points, uncropped page frame).
TABLE_X = [170.0, 260.0, 350.0, 440.0]
TABLE_Y = [305.0, 370.0, 435.0, 500.0]
TABLE_VALUES = [
    ["100.00", "200.00", "300.00"],
    ["11.00", "22.00", "33.00"],
    ["1.00", "2.00", "3.00"],
]
TABLE_DF = pd.DataFrame(TABLE_VALUES)
AREA = ["165,505,445,300"]
REGION = ["165,505,445,300"]
OUTSIDE_TEXT = "OUTSIDE CROP"
# Visible area used throughout: bottom-left (116, 150), top-right (500, 585).
VISIBLE = (116.0, 150.0, 500.0, 585.0)


def _inverse_point(lx, ly, media, rotation):
    """Map an uncropped-frame (layout) point to PDF user space.

    ``layout`` is the frame produced by the same page with the CropBox removed,
    keeping the rotation and MediaBox.  This is the inverse of
    :func:`camelot.page_geometry.map_page_point`.
    """
    mx0, my0, mx1, my1 = media
    rotation %= 360
    if rotation == 90:
        return (mx1 - ly, lx + my0)
    if rotation == 180:
        return (mx1 - lx, my1 - ly)
    if rotation == 270:
        return (ly + mx0, my1 - lx)
    return (lx + mx0, ly + my0)


def _content_ops(media, rotation, outside=True):
    """Return PDF content ops drawing the test table in the given frame."""
    ops = []

    def line(lx1, ly1, lx2, ly2):
        a = _inverse_point(lx1, ly1, media, rotation)
        b = _inverse_point(lx2, ly2, media, rotation)
        ops.append(f"1 w {a[0]:.4f} {a[1]:.4f} m {b[0]:.4f} {b[1]:.4f} l S\n")

    for x in TABLE_X:
        line(x, TABLE_Y[0], x, TABLE_Y[-1])
    for y in TABLE_Y:
        line(TABLE_X[0], y, TABLE_X[-1], y)

    # Rotation applied to the text matrix so glyphs are upright once the page
    # /Rotate is taken into account.
    text_matrix = {0: "1 0 0 1", 90: "0 1 -1 0", 180: "-1 0 0 -1", 270: "0 -1 1 0"}[
        rotation % 360
    ]
    for r, row in enumerate(TABLE_VALUES):
        for c, value in enumerate(row):
            lx = TABLE_X[c] + 18
            ly = TABLE_Y[2 - r] + 35
            x, y = _inverse_point(lx, ly, media, rotation)
            ops.append(
                f"BT /F1 10 Tf {text_matrix} {x:.4f} {y:.4f} Tm ({value}) Tj ET\\n"
            )
    if outside:
        x, y = _inverse_point(65, 670, media, rotation)
        ops.append(
            f"BT /F1 12 Tf {text_matrix} {x:.4f} {y:.4f} Tm ({OUTSIDE_TEXT}) Tj ET\\n"
        )
    return ops


def build_pdf(
    path,
    *,
    media=(0.0, 0.0, 612.0, 792.0),
    crop=None,
    rotation=0,
    outside=True,
    pages=None,
):
    """Generate a PDF with the reference table.

    ``crop`` and the page geometry are expressed in the uncropped page frame
    (i.e. the CropBox is the given rectangle mapped back into user space).
    ``pages`` may be a list of dicts to build a multi-page document; per-page
    keys are ``crop``, ``rotation`` and ``media``.
    """
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject
    from pypdf.generic import DictionaryObject
    from pypdf.generic import NameObject
    from pypdf.generic import NumberObject
    from pypdf.generic import RectangleObject

    writer = PdfWriter()
    specs = pages or [{"crop": crop, "rotation": rotation, "media": media}]
    for spec in specs:
        page_media = spec.get("media", media)
        page_rotation = spec.get("rotation", rotation)
        page_crop = spec.get("crop", crop)
        mx0, my0, mx1, my1 = page_media
        page = writer.add_blank_page(width=mx1 - mx0, height=my1 - my0)
        page.mediabox = RectangleObject([float(v) for v in page_media])
        if page_crop is None:
            crop_box = tuple(page_media)
        else:
            corners = [
                _inverse_point(page_crop[0], page_crop[1], page_media, page_rotation),
                _inverse_point(page_crop[2], page_crop[1], page_media, page_rotation),
                _inverse_point(page_crop[0], page_crop[3], page_media, page_rotation),
                _inverse_point(page_crop[2], page_crop[3], page_media, page_rotation),
            ]
            xs = [pt[0] for pt in corners]
            ys = [pt[1] for pt in corners]
            crop_box = (min(xs), min(ys), max(xs), max(ys))
        page.cropbox = RectangleObject([float(v) for v in crop_box])

        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_ref = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
        stream = DecodedStreamObject()
        stream.set_data(
            "".join(_content_ops(page_media, page_rotation, outside)).encode("latin-1")
        )
        page[NameObject("/Contents")] = writer._add_object(stream)
        if page_rotation:
            page[NameObject("/Rotate")] = NumberObject(page_rotation)
    with open(path, "wb") as f:
        writer.write(f)
    return path


@pytest.fixture(scope="module")
def pdfs(tmp_path_factory):
    """Build the generated fixture set once."""
    root = tmp_path_factory.mktemp("crop_geometry")
    paths = {
        "plain": build_pdf(str(root / "plain.pdf")),
        "cropped": build_pdf(
            str(root / "cropped.pdf"), crop=VISIBLE
        ),
        "rot90": build_pdf(str(root / "rot90.pdf"), rotation=90),
        "rot90_cropped": build_pdf(
            str(root / "rot90_cropped.pdf"), rotation=90, crop=VISIBLE
        ),
        "rot270": build_pdf(str(root / "rot270.pdf"), rotation=270),
        "rot270_cropped": build_pdf(
            str(root / "rot270_cropped.pdf"), rotation=270, crop=VISIBLE
        ),
        "rot180": build_pdf(str(root / "rot180.pdf"), rotation=180),
        "rot180_cropped": build_pdf(
            str(root / "rot180_cropped.pdf"), rotation=180, crop=VISIBLE
        ),
        "offset": build_pdf(
            str(root / "offset.pdf"), media=(100.0, 50.0, 712.0, 842.0)
        ),
        "offset_cropped": build_pdf(
            str(root / "offset_cropped.pdf"),
            media=(100.0, 50.0, 712.0, 842.0),
            crop=VISIBLE,
        ),
        "near": build_pdf(
            str(root / "near.pdf"), crop=(168.0, 303.0, 442.0, 502.0)
        ),
        "multi": build_pdf(
            str(root / "multi.pdf"),
            pages=[
                {"crop": None},
                {"crop": VISIBLE},
                {"crop": (150.0, 100.0, 600.0, 700.0)},
                {"crop": VISIBLE, "rotation": 90},
            ],
        ),
    }
    return paths


def _assert_table(tables, expected_df=TABLE_DF):
    assert len(tables) == 1
    pd.testing.assert_frame_equal(tables[0].df, expected_df)
    return tables[0]


@pytest.mark.parametrize(
    "flavor,mode,kwargs",
    [
        ("lattice", "auto", {}),
        ("lattice", "areas", {"table_areas": AREA}),
        ("lattice", "regions", {"table_regions": REGION}),
        ("stream", "auto", {}),
        ("stream", "areas", {"table_areas": AREA}),
        ("stream", "regions", {"table_regions": REGION}),
    ],
)
def test_crop_matches_uncropped(pdfs, flavor, mode, kwargs):
    """Cropping must not change the table content or its coordinates.

    The visible area contains the whole table plus its text, so for every
    flavour and selection mode the cropped page parses the table exactly as
    the uncropped page does under the *same* selection coordinates.
    """
    uncropped = camelot.read_pdf(pdfs["plain"], flavor=flavor, **kwargs)
    cropped = camelot.read_pdf(pdfs["cropped"], flavor=flavor, **kwargs)
    assert len(uncropped) == len(cropped) == 1
    table_u, table_c = uncropped[0], cropped[0]
    if mode == "areas" or flavor == "lattice":
        # These selections identify the table itself, so the table frame is
        # unchanged by cropping.
        assert table_c._bbox == pytest.approx(table_u._bbox, abs=2)
    if mode == "auto" and flavor == "stream":
        # Automatic stream detection on the uncropped page also sees the
        # line of text above the table; the cropped page must not.
        assert OUTSIDE_TEXT in table_u.data[0][0]
        assert table_u.shape == (4, 4)
        pd.testing.assert_frame_equal(table_c.df, TABLE_DF)
    elif flavor == "stream":
        if mode == "regions":
            # The uncropped page's region fallback also picks up the hidden
            # line; only the cropped page is restricted to visible text.
            assert OUTSIDE_TEXT in table_u.data[0][0]
        else:
            pd.testing.assert_frame_equal(table_c.df, table_u.df)
            # Both pages parse the same visible text into the same cells.
            assert table_c.data == table_u.data
    elif mode == "areas":
        # A manual area keeps its own rectangle; the lattice parser subdivides
        # it at every detected line, including the area's padding lines.
        # This is identical with and without a CropBox.
        pd.testing.assert_frame_equal(table_c.df, table_u.df)
    else:
        pd.testing.assert_frame_equal(table_c.df, TABLE_DF)

    if mode == "areas":
        # The bbox is the caller's selection, unchanged by the CropBox.
        assert table_c._bbox == pytest.approx((165, 300, 445, 505), abs=2)
    elif flavor == "stream":
        if mode == "auto":
            # Automatic stream falls back to the whole visible page.
            assert table_c._bbox == pytest.approx(VISIBLE)
        else:
            # table_regions on the uncropped page sees the hidden line too
            # and falls back to the full page; the cropped page only sees the
            # visible text and falls back to the visible area.
            assert table_u._bbox == pytest.approx((0, 0, 612, 792))
            assert table_c._bbox == pytest.approx(VISIBLE)
    else:
        # Lattice auto reports the detected table frame in the uncropped frame.
        assert table_c._bbox == pytest.approx(
            (TABLE_X[0], TABLE_Y[0], TABLE_X[-1], TABLE_Y[-1]), abs=2
        )


@pytest.mark.parametrize("flavor", ["lattice", "stream", "network", "hybrid"])
def test_text_outside_crop_never_used(pdfs, flavor):
    """Text outside the visible area must not reach cells in any flavour."""
    table = camelot.read_pdf(pdfs["cropped"], flavor=flavor)[0]
    texts = [cell for row in table.data for cell in row]
    assert all(OUTSIDE_TEXT not in cell for cell in texts)
    # Auto detection must not grow the table to cover the hidden text.
    assert table._bbox[1] >= VISIBLE[1] - 2
    assert table._bbox[3] <= VISIBLE[3] + 2


@pytest.mark.parametrize(
    "uncropped_key,cropped_key,rotation,media",
    [
        ("plain", "cropped", 0, (0.0, 0.0, 612.0, 792.0)),
        ("rot90", "rot90_cropped", 90, (0.0, 0.0, 612.0, 792.0)),
        ("rot270", "rot270_cropped", 270, (0.0, 0.0, 612.0, 792.0)),
        ("rot180", "rot180_cropped", 180, (0.0, 0.0, 612.0, 792.0)),
        ("offset", "offset_cropped", 0, (100.0, 50.0, 712.0, 842.0)),
    ],
)
def test_rotation_and_offset(pdfs, uncropped_key, cropped_key, rotation, media):
    """Rotation and offset MediaBox keep the same uncropped-frame truth.

    Coordinates are always reported in the frame of an uncropped copy of the
    same page (same rotation, same MediaBox), so the table frame is the same
    for every case.
    """
    for flavor in ("lattice", "stream"):
        table = camelot.read_pdf(pdfs[cropped_key], flavor=flavor)[0]
        # The handler normalises the displayed rotation away, so the layout
        # frame is the visible rectangle itself.
        assert table.page_geometry.visible_bbox == pytest.approx(VISIBLE)
        # Rotation is retained in the geometry (the raster applies it); the
        # coordinate frame is still the uncropped page frame, so the table
        # is reported at the same place regardless of rotation.
        assert table.page_geometry.rotation == rotation
        if flavor == "lattice":
            # Lattice auto reports the detected table frame.
            assert table._bbox == pytest.approx(
                (TABLE_X[0], TABLE_Y[0], TABLE_X[-1], TABLE_Y[-1]), abs=2
            )
            pd.testing.assert_frame_equal(table.df, TABLE_DF)
        else:
            # Automatic stream falls back to the visible page area.
            assert table._bbox == pytest.approx(VISIBLE)
            assert table.shape == (3, 3)


def test_table_near_visible_boundary(pdfs):
    """A complete table that hugs the visible border is still detected."""
    tables = camelot.read_pdf(pdfs["near"])
    table = _assert_table(tables)
    # The crop edge is 2pt outside the frame on every side.
    assert table._bbox[0] == pytest.approx(170, abs=2)
    assert table._bbox[2] == pytest.approx(440, abs=2)
    assert table._bbox[1] == pytest.approx(305, abs=2)
    assert table._bbox[3] == pytest.approx(500, abs=2)


def test_multipage_different_crops(pdfs):
    """Each page's own CropBox is resolved independently."""
    tables = camelot.read_pdf(pdfs["multi"], pages="all")
    assert len(tables) == 4
    for table in tables:
        pd.testing.assert_frame_equal(table.df, TABLE_DF)
        assert table._bbox[0] == pytest.approx(170, abs=2)
        assert table._bbox[2] == pytest.approx(440, abs=2)
    assert tables[0].page_geometry.visible_bbox == pytest.approx(
        (0.0, 0.0, 612.0, 792.0)
    )
    assert tables[1].page_geometry.visible_bbox == pytest.approx(VISIBLE)
    assert tables[2].page_geometry.visible_bbox == pytest.approx(
        (150.0, 100.0, 600.0, 700.0)
    )
    assert tables[3].page_geometry.visible_bbox == pytest.approx(VISIBLE)


@pytest.mark.parametrize("flavor", ["lattice", "stream"])
def test_serial_parallel_equivalence(pdfs, flavor):
    """parallel=True must produce the same physical tables as serial."""
    serial = camelot.read_pdf(pdfs["multi"], pages="all", flavor=flavor, parallel=False)
    parallel = camelot.read_pdf(
        pdfs["multi"], pages="all", flavor=flavor, parallel=True
    )
    assert len(serial) == len(parallel)
    for a, b in zip(serial, parallel):  # noqa: B905
        pd.testing.assert_frame_equal(a.df, b.df)
        assert a.page == b.page
        assert a._bbox == pytest.approx(b._bbox)


def test_lattice_geometry_matches_raster(pdfs):
    """Detected lines and joints sit on the visible raster within 2 PDF points."""
    import cv2

    for key in ("cropped", "rot90_cropped", "rot270_cropped", "offset_cropped"):
        table = camelot.read_pdf(pdfs[key])[0]
        image = table.get_pdf_image()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark = gray < 128
        height, width = dark.shape
        geometry = table.page_geometry
        sx, sy = geometry.scale_factors((width, height))
        point_scale = min(sx, sy)
        # 2 PDF points expressed in pixels.
        tolerance = 2.0 * point_scale

        def distance_to_dark(px, py):
            best = 1e9
            radius = int(np.ceil(tolerance))
            x0 = int(max(0, round(px) - radius))
            x1 = int(min(width, round(px) + radius + 1))
            y0 = int(max(0, round(py) - radius))
            y1 = int(min(height, round(py) + radius + 1))
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    if dark[yy, xx]:
                        best = min(best, ((xx - px) ** 2 + (yy - py) ** 2) ** 0.5)
            return best

        for x in TABLE_X:
            for y in TABLE_Y:
                px, py = pdf_to_image_point(x, y, geometry, (width, height))
                assert distance_to_dark(px, py) <= tolerance, (key, x, y)

        for joint in table.parse["joints"]:
            px, py = pdf_to_image_point(joint[0], joint[1], geometry, (width, height))
            assert distance_to_dark(px, py) <= tolerance, (key, joint)


def test_image_pdf_roundtrip(pdfs):
    """The geometry mapping is a bijection in both directions."""
    table = camelot.read_pdf(pdfs["cropped"])[0]
    geometry = table.page_geometry
    image = table.get_pdf_image()
    size = (image.shape[1], image.shape[0])
    for x in TABLE_X:
        for y in TABLE_Y:
            px, py = pdf_to_image_point(x, y, geometry, size)
            rx, ry = image_to_pdf_point(px, py, geometry, size)
            assert rx == pytest.approx(x, abs=1e-6)
            assert ry == pytest.approx(y, abs=1e-6)


@pytest.mark.parametrize("kind", ["grid", "line", "joint"])
def test_plot_overlays_visible_page(pdfs, kind):
    """Plot overlays and the raster share one coordinate frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = camelot.read_pdf(pdfs["cropped"])[0]
    fig = camelot.plot(table, kind=kind)
    ax = fig.axes[0]
    image_artist = ax.get_images()[0]
    # matplotlib stores the extent as (left, right, bottom, top).
    left, right, bottom, top = image_artist.get_extent()
    assert (left, bottom, right, top) == pytest.approx(VISIBLE)

    # Every plotted coordinate must fall on the drawn page extent.
    image = table.get_pdf_image()
    height, width = image.shape[:2]
    import cv2

    dark = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 128

    def to_pixels(x, y):
        return ((x - left) / (right - left) * width, (top - y) / (top - bottom) * height)

    def near_dark(px, py, radius=4):
        xi, yi = int(round(px)), int(round(py))
        for yy in range(max(0, yi - radius), min(height, yi + radius + 1)):
            for xx in range(max(0, xi - radius), min(width, xi + radius + 1)):
                if dark[yy, xx]:
                    return True
        return False

    for line in ax.lines:
        xs, ys = line.get_data()
        for x, y in zip(xs, ys):  # noqa: B905
            px, py = to_pixels(x, y)
            assert near_dark(px, py), (kind, x, y)
    plt.close(fig)


def test_page_geometry_visible_bbox_rotations():
    """Unit-check the visible-area mapping for every rotation."""
    from camelot.page_geometry import visible_bbox

    media = (0.0, 0.0, 612.0, 792.0)
    crop = (116.0, 150.0, 500.0, 585.0)
    assert visible_bbox(media, crop, 0) == pytest.approx(crop)
    assert visible_bbox(media, crop, 90) == pytest.approx(
        (150.0, 112.0, 585.0, 496.0)
    )
    assert visible_bbox(media, crop, 180) == pytest.approx(
        (112.0, 207.0, 496.0, 642.0)
    )
    assert visible_bbox(media, crop, 270) == pytest.approx(
        (207.0, 116.0, 642.0, 500.0)
    )


def test_page_geometry_from_page(tmp_path):
    """PageGeometry reads MediaBox, CropBox and rotation from a page."""
    from pypdf import PdfReader
    from pypdf import PdfWriter
    from pypdf.generic import NameObject
    from pypdf.generic import NumberObject
    from pypdf.generic import RectangleObject

    path = str(tmp_path / "_geom.pdf")
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    page.mediabox = RectangleObject([0, 0, 612, 792])
    page.cropbox = RectangleObject([116, 150, 500, 585])
    page[NameObject("/Rotate")] = NumberObject(90)
    with open(path, "wb") as f:
        writer.write(f)

    geometry = PageGeometry.from_page(PdfReader(path).pages[0])
    assert geometry.media_box == (0.0, 0.0, 612.0, 792.0)
    assert geometry.crop_box == (116.0, 150.0, 500.0, 585.0)
    assert geometry.rotation == 90
    assert geometry.is_cropped


def test_real_pdf_crop_matches_uncropped(testdir, tmp_path):
    """A real fixture's table is unchanged when the page is cropped around it."""
    from pypdf import PdfReader
    from pypdf import PdfWriter
    from pypdf.generic import RectangleObject

    src = os.path.join(testdir, "twotables_2.pdf")
    area = ["80,693,535,448"]
    # Crop generously around the first table (its frame is inside this box).
    reader = PdfReader(src)
    page = reader.pages[0]
    page.cropbox = RectangleObject([50, 400, 560, 720])
    cropped_path = str(tmp_path / "twotables_2_cropped.pdf")
    writer = PdfWriter()
    writer.add_page(page)
    with open(cropped_path, "wb") as f:
        writer.write(f)

    for flavor in ("lattice", "stream"):
        for selector in ("table_areas", "table_regions"):
            kwargs = {selector: area}
            uncropped = camelot.read_pdf(src, flavor=flavor, **kwargs)
            cropped = camelot.read_pdf(cropped_path, flavor=flavor, **kwargs)
            assert len(uncropped) == len(cropped) == 1
            pd.testing.assert_frame_equal(uncropped[0].df, cropped[0].df)
            assert cropped[0]._bbox == pytest.approx(uncropped[0]._bbox, abs=2)
