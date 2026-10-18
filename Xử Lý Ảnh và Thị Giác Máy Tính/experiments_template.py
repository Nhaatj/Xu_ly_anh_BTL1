"""Experiment and plotting harness for student-implemented image algorithms.

Connect the adapters below to your own implementations before running this file.
This harness does not implement CLAHE, adaptive thresholding, Canny, or Hough.
"""

from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Set this to the grayscale image used in your assignment.
IMAGE_PATH = Path(r"data\input.png")
OUTPUT_DIR = Path("experiment_results")

# Expected adapter interfaces:
#   run_clahe(image, grid, clip_limit) -> grayscale image, same HxW shape
#   run_adaptive(image, method, window_size, k, R, offset) -> binary image, same HxW shape
#   run_canny(image, low_threshold, high_threshold) -> (edge_image, gradient_angle_radians)
#   run_hough(edge_image, gradient_angle_radians, parameters) -> iterable of (rho, theta)
#
# Replace each None with a small wrapper around your own functions. The wrapper
# is for adapting argument names/return formats only; do not put algorithm code here.
run_clahe: Callable | None = None
run_adaptive: Callable | None = None
run_canny: Callable | None = None
run_hough: Callable | None = None

CLAHE_CONFIGS = [
    {"grid": (4, 4), "clip_limit": 1.0},
    {"grid": (8, 8), "clip_limit": 2.0},
    {"grid": (8, 8), "clip_limit": 4.0},
    {"grid": (12, 12), "clip_limit": 2.0},
]

ADAPTIVE_CONFIGS = [
    {"method": "bradley", "window_size": 31, "k": 0.15, "R": 128.0, "offset": 0.0},
    {"method": "bradley", "window_size": 51, "k": 0.20, "R": 128.0, "offset": 0.0},
    {"method": "sauvola", "window_size": 31, "k": 0.20, "R": 128.0, "offset": 0.0},
    {"method": "sauvola", "window_size": 51, "k": 0.35, "R": 128.0, "offset": 0.0},
]

# Adapt these dictionaries to the parameter names accepted by your Hough wrapper.
HOUGH_CONFIGS = [
    {"rho_step": 1.0, "theta_step_degrees": 2.0, "delta_degrees": 5.0, "top_k": 10},
    {"rho_step": 1.0, "theta_step_degrees": 1.0, "delta_degrees": 5.0, "top_k": 10},
]

CANNY_CONFIG = {"low_threshold": 40.0, "high_threshold": 100.0}


def load_grayscale(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path.resolve()}")
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def require_adapter(name: str, adapter: Callable | None) -> Callable:
    if adapter is None:
        raise RuntimeError(
            f"Set {name} to a wrapper for your own implementation before running."
        )
    return adapter


def require_image_result(result: np.ndarray, expected_shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(result)
    if array.shape != expected_shape:
        raise ValueError(f"{label} returned shape {array.shape}; expected {expected_shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} returned NaN or infinite values.")
    return array


def save_grid(
    source: np.ndarray,
    results: Sequence[tuple[str, np.ndarray]],
    title: str,
    output_path: Path,
    cmap: str = "gray",
) -> None:
    count = len(results) + 1
    columns = min(3, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes_array = np.atleast_1d(axes).ravel()

    axes_array[0].imshow(source, cmap="gray", vmin=0, vmax=255)
    axes_array[0].set_title("Input")
    axes_array[0].axis("off")
    for axis, (label, result) in zip(axes_array[1:], results):
        axis.imshow(result, cmap=cmap)
        axis.set_title(label)
        axis.axis("off")
    for axis in axes_array[count:]:
        axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path.resolve()}")


def run_clahe_experiments(image: np.ndarray) -> None:
    adapter = require_adapter("run_clahe", run_clahe)
    results = []
    for config in CLAHE_CONFIGS:
        output = adapter(image, config["grid"], config["clip_limit"])
        output = require_image_result(output, image.shape, f"CLAHE {config}")
        label = f"grid={config['grid']}, clip={config['clip_limit']}"
        results.append((label, output))
    save_grid(image, results, "CLAHE parameter comparison", OUTPUT_DIR / "clahe.png")


def run_adaptive_experiments(image: np.ndarray) -> None:
    adapter = require_adapter("run_adaptive", run_adaptive)
    results = []
    for config in ADAPTIVE_CONFIGS:
        output = adapter(
            image,
            config["method"],
            config["window_size"],
            config["k"],
            config["R"],
            config["offset"],
        )
        output = require_image_result(output, image.shape, f"Adaptive threshold {config}")
        label = (
            f"{config['method']}, window={config['window_size']}, "
            f"k={config['k']}, offset={config['offset']}"
        )
        results.append((label, output))
    save_grid(image, results, "Adaptive threshold parameter comparison", OUTPUT_DIR / "adaptive.png")


def line_endpoints(rho: float, theta: float, width: int, height: int) -> list[tuple[float, float]]:
    """Return image-border intersections for plotting a Hough line."""
    cosine, sine = np.cos(theta), np.sin(theta)
    candidates: list[tuple[float, float]] = []
    if abs(sine) > 1e-12:
        candidates.extend([(0.0, rho / sine), (width - 1.0, (rho - (width - 1) * cosine) / sine)])
    if abs(cosine) > 1e-12:
        candidates.extend([(rho / cosine, 0.0), ((rho - (height - 1) * sine) / cosine, height - 1.0)])

    valid: list[tuple[float, float]] = []
    for x, y in candidates:
        if 0 <= x < width and 0 <= y < height and all(
            np.hypot(x - old_x, y - old_y) > 1e-6 for old_x, old_y in valid
        ):
            valid.append((x, y))
    return valid[:2]


def run_hough_experiments(image: np.ndarray) -> None:
    canny = require_adapter("run_canny", run_canny)
    hough = require_adapter("run_hough", run_hough)
    edges, gradient = canny(image, CANNY_CONFIG["low_threshold"], CANNY_CONFIG["high_threshold"])
    edges = require_image_result(edges, image.shape, "Canny")
    gradient = require_image_result(gradient, image.shape, "Canny gradient")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for index, config in enumerate(HOUGH_CONFIGS, start=1):
        peaks: Iterable[tuple[float, float]] = hough(edges, gradient, config)
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        for rho, theta in peaks:
            endpoints = line_endpoints(float(rho), float(theta), image.shape[1], image.shape[0])
            if len(endpoints) == 2:
                (x1, y1), (x2, y2) = endpoints
                axis.plot([x1, x2], [y1, y2], linewidth=1.2)
        axis.set_title(f"Hough peaks: {config}")
        axis.axis("off")
        output_path = OUTPUT_DIR / f"hough_{index}.png"
        figure.tight_layout()
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {output_path.resolve()}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = load_grayscale(IMAGE_PATH)
    run_clahe_experiments(image)
    run_adaptive_experiments(image)
    run_hough_experiments(image)


if __name__ == "__main__":
    main()
