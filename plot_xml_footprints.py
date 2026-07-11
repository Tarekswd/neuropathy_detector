"""Generate max-pressure PNGs and rollover GIFs from fixed_xml/ Zebris files."""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from plot_common import PROJECT_ROOT, XML_NS, ensure_dir, setup_matplotlib, plot_pressure_heatmap

XML_DIR = PROJECT_ROOT / "fixed_xml"
OUTPUT_DIR = PROJECT_ROOT / "output_plots"
COLORMAP = plt.cm.hot


def array_to_rgb(array: np.ndarray, vmax: float) -> np.ndarray:
    normalized = np.clip(array / max(vmax, 1e-6), 0, 1)
    rgba = COLORMAP(normalized)
    rgb = np.zeros((*array.shape, 3), dtype=np.uint8)
    rgb[...] = (rgba[..., :3] * 255).astype(np.uint8)
    rgb[array <= 0] = 0
    return rgb


def parse_cells(cells_text: str, width: int, height: int) -> np.ndarray:
    values = [float(value) for value in cells_text.split()]
    expected = width * height
    if len(values) != expected:
        raise ValueError(f"Expected {expected} cells, got {len(values)}")
    return np.array(values, dtype=float).reshape(height, width)


def assemble_quant_frame(quant_element, canvas_width: int, canvas_height: int) -> np.ndarray:
    frame = np.zeros((canvas_height, canvas_width), dtype=float)
    cell_begin = quant_element.find("z:cell_begin", XML_NS)
    cell_count = quant_element.find("z:cell_count", XML_NS)
    cells = quant_element.find("z:cells", XML_NS)

    x0 = int(cell_begin.find("z:x", XML_NS).text)
    y0 = int(cell_begin.find("z:y", XML_NS).text)
    width = int(cell_count.find("z:x", XML_NS).text)
    height = int(cell_count.find("z:y", XML_NS).text)
    block = parse_cells(cells.text, width, height)
    frame[y0 : y0 + height, x0 : x0 + width] = block
    return frame


def parse_event(event_element) -> dict:
    rollover = event_element.find("z:rollover", XML_NS)
    max_block = event_element.find("z:max", XML_NS)

    canvas_width = int(rollover.find("z:cell_count/z:x", XML_NS).text)
    canvas_height = int(rollover.find("z:cell_count/z:y", XML_NS).text)

    max_width = int(max_block.find("z:cell_count/z:x", XML_NS).text)
    max_height = int(max_block.find("z:cell_count/z:y", XML_NS).text)
    max_pressure = parse_cells(max_block.find("z:cells", XML_NS).text, max_width, max_height)

    frames = [
        assemble_quant_frame(quant, canvas_width, canvas_height)
        for quant in rollover.findall(".//z:quant", XML_NS)
    ]

    return {
        "event_id": event_element.find("z:id", XML_NS).text,
        "side": event_element.find("z:side", XML_NS).text,
        "max_pressure": max_pressure,
        "frames": frames,
    }


def save_max_pressure_png(event_data: dict, xml_stem: str, output_dir: Path) -> Path:
    pressure = event_data["max_pressure"]
    output_path = output_dir / (
        f"{xml_stem}_{event_data['event_id']}_{event_data['side']}_max_pressure.png"
    )
    fig, ax = plt.subplots(figsize=(5, 8), constrained_layout=True)
    fig.patch.set_facecolor("black")
    plot_pressure_heatmap(
        ax,
        pressure,
        title=f"{xml_stem} {event_data['event_id']} {event_data['side']} max pressure",
        vmax=float(pressure.max())
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_rollover_gif(event_data: dict, xml_stem: str, output_dir: Path, duration_ms: int = 80) -> Path:
    frames = event_data["frames"]
    if not frames:
        raise ValueError(f"No rollover frames for {xml_stem} {event_data['event_id']}")

    vmax = max(float(frame.max()) for frame in frames)
    images = []
    for frame in frames:
        img = Image.fromarray(array_to_rgb(frame, vmax))
        # Upscale the image to make it visible
        new_size = (img.width * 10, img.height * 10)
        img = img.resize(new_size, Image.Resampling.NEAREST)
        images.append(img)
    output_path = output_dir / (
        f"{xml_stem}_{event_data['event_id']}_{event_data['side']}_rollover.gif"
    )
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path


def process_xml(xml_path: Path, output_dir: Path) -> list[Path]:
    root = ET.parse(xml_path).getroot()
    saved_paths: list[Path] = []
    for event_element in root.findall(".//z:event", XML_NS):
        event_data = parse_event(event_element)
        saved_paths.append(save_max_pressure_png(event_data, xml_path.stem, output_dir))
        saved_paths.append(save_rollover_gif(event_data, xml_path.stem, output_dir))
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot XML pressure maps and rollover animations.")
    parser.add_argument("--xml-dir", type=Path, default=XML_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of XML files.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    setup_matplotlib()
    xml_files = sorted(args.xml_dir.glob("*.xml"))
    if args.max_files is not None:
        xml_files = xml_files[: args.max_files]
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {args.xml_dir}")

    total_outputs = 0
    for xml_path in xml_files:
        outputs = process_xml(xml_path, args.output_dir)
        total_outputs += len(outputs)
        print(f"{xml_path.name}: saved {len(outputs)} files", flush=True)

    print(f"Done. {total_outputs} files written to {args.output_dir}")


if __name__ == "__main__":
    main()
