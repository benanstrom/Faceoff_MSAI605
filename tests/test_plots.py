from __future__ import annotations

from pathlib import Path

from PIL import Image

from lfw_verif.plots import plot_confusion_matrix, plot_roc_curve


def test_plot_roc_curve_writes_png_file(tmp_path: Path) -> None:
    output_path = tmp_path / "roc.png"

    written = plot_roc_curve([0.95, 0.7, 0.4, 0.1], [1, 0, 1, 0], output_path)

    assert written == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    with Image.open(output_path) as image:
        assert image.format == "PNG"


def test_plot_confusion_matrix_writes_png_file(tmp_path: Path) -> None:
    output_path = tmp_path / "confusion_matrix.png"

    written = plot_confusion_matrix([0.95, 0.7, 0.4, 0.1], [1, 0, 1, 0], 0.5, output_path)

    assert written == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    with Image.open(output_path) as image:
        assert image.format == "PNG"
