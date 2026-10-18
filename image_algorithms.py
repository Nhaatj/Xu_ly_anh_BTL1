"""NumPy implementations of CLAHE, integral-image thresholding, and Hough."""

import numpy as np


def _gray_uint8(img, name="img"):
    array = np.asarray(img)
    if array.ndim != 2 or array.dtype != np.uint8:
        raise ValueError(f"{name} must be a 2D uint8 grayscale image.")
    return array


def clip_histogram(hist, clip_limit):
    """Clip a histogram and redistribute excess while preserving its sum."""
    counts = np.asarray(hist)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("hist must be a one-dimensional integer array.")
    if not np.isfinite(clip_limit) or clip_limit < 0:
        raise ValueError("clip_limit must be finite and non-negative.")

    clipped = counts.astype(np.int64, copy=True)
    limit = max(1, int(np.floor(clip_limit)))
    excess = int(np.maximum(clipped - limit, 0).sum())
    clipped = np.minimum(clipped, limit)
    if clipped.size == 0:
        return clipped

    quotient, remainder = divmod(excess, clipped.size)
    clipped += quotient
    if remainder:
        indices = np.linspace(0, clipped.size - 1, remainder, dtype=np.int64)
        clipped[indices] += 1
    return clipped


def build_tile_luts(img, grid=(8, 8), clip_limit=2.0):
    """Build per-tile uint8 LUTs; clip_limit is normalized as s_max."""
    image = _gray_uint8(img)
    gy, gx = map(int, grid)
    if gy <= 0 or gx <= 0 or gy > image.shape[0] or gx > image.shape[1]:
        raise ValueError("grid must be positive and no larger than the image.")
    if not np.isfinite(clip_limit) or clip_limit < 0:
        raise ValueError("clip_limit must be finite and non-negative.")

    y_edges = np.linspace(0, image.shape[0], gy + 1, dtype=np.int64)
    x_edges = np.linspace(0, image.shape[1], gx + 1, dtype=np.int64)
    luts = np.empty((gy, gx, 256), dtype=np.uint8)
    for i in range(gy):
        for j in range(gx):
            tile = image[y_edges[i]:y_edges[i + 1], x_edges[j]:x_edges[j + 1]]
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.int64)
            beta = max(1, int(np.floor(clip_limit * tile.size / 256)))
            hist = clip_histogram(hist, beta)
            cdf = np.cumsum(hist, dtype=np.int64)
            luts[i, j] = np.clip(
                np.rint(255.0 * cdf / tile.size), 0, 255
            ).astype(np.uint8)
    return luts


def clahe(img, grid=(8, 8), clip_limit=2.0):
    """Apply CLAHE using per-tile LUTs and bilinear interpolation."""
    image = _gray_uint8(img)
    gy, gx = map(int, grid)
    luts = build_tile_luts(image, (gy, gx), clip_limit)

    y_edges = np.linspace(0, image.shape[0], gy + 1, dtype=np.int64)
    x_edges = np.linspace(0, image.shape[1], gx + 1, dtype=np.int64)
    y_centers = (y_edges[:-1] + y_edges[1:] - 1) / 2.0
    x_centers = (x_edges[:-1] + x_edges[1:] - 1) / 2.0
    ys = np.arange(image.shape[0], dtype=np.float64)
    xs = np.arange(image.shape[1], dtype=np.float64)

    y0 = np.clip(np.searchsorted(y_centers, ys, side="right") - 1, 0, gy - 1)
    x0 = np.clip(np.searchsorted(x_centers, xs, side="right") - 1, 0, gx - 1)
    y1 = np.minimum(y0 + 1, gy - 1)
    x1 = np.minimum(x0 + 1, gx - 1)
    y_den = y_centers[y1] - y_centers[y0]
    x_den = x_centers[x1] - x_centers[x0]
    b = np.divide(
        ys - y_centers[y0], y_den, out=np.zeros_like(ys), where=y_den > 0
    )
    a = np.divide(
        xs - x_centers[x0], x_den, out=np.zeros_like(xs), where=x_den > 0
    )
    a = np.clip(a, 0.0, 1.0)[None, :]
    b = np.clip(b, 0.0, 1.0)[:, None]
    values = image.astype(np.intp)

    t00 = luts[y0[:, None], x0[None, :], values]
    t01 = luts[y0[:, None], x1[None, :], values]
    t10 = luts[y1[:, None], x0[None, :], values]
    t11 = luts[y1[:, None], x1[None, :], values]
    output = (1 - b) * ((1 - a) * t00 + a * t01) + b * (
        (1 - a) * t10 + a * t11
    )
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def integral_image(img):
    """Return an int64 summed-area table with a zero-padded top row/column."""
    values = np.asarray(img, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("img must be a two-dimensional array.")
    summed = np.zeros((values.shape[0] + 1, values.shape[1] + 1), dtype=np.int64)
    summed[1:, 1:] = np.cumsum(
        np.cumsum(values, axis=0, dtype=np.int64), axis=1, dtype=np.int64
    )
    return summed


def box_sum(S, y1, x1, y2, x2):
    """Return sums over inclusive boxes using a padded summed-area table."""
    table = np.asarray(S)
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("S must be a 2D summed-area table with a zero border.")
    y1, x1, y2, x2 = np.broadcast_arrays(y1, x1, y2, x2)
    if np.any(y1 < 0) or np.any(x1 < 0) or np.any(y2 < y1) or np.any(x2 < x1):
        raise ValueError("Invalid box coordinates.")
    if np.any(y2 >= table.shape[0] - 1) or np.any(x2 >= table.shape[1] - 1):
        raise ValueError("Box coordinates are outside the image.")
    return (
        table[y2 + 1, x2 + 1]
        - table[y1, x2 + 1]
        - table[y2 + 1, x1]
        + table[y1, x1]
    )


def local_mean_std(img, radius):
    """Compute local mean/std using two summed-area tables."""
    image = _gray_uint8(img).astype(np.int64)
    if not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("radius must be a non-negative integer.")
    height, width = image.shape
    ys = np.arange(height)
    xs = np.arange(width)
    y1 = np.maximum(ys - radius, 0)[:, None]
    y2 = np.minimum(ys + radius, height - 1)[:, None]
    x1 = np.maximum(xs - radius, 0)[None, :]
    x2 = np.minimum(xs + radius, width - 1)[None, :]

    sum_image = box_sum(integral_image(image), y1, x1, y2, x2).astype(np.float64)
    sum_squared = box_sum(
        integral_image(image * image), y1, x1, y2, x2
    ).astype(np.float64)
    count = (y2 - y1 + 1) * (x2 - x1 + 1)
    mean = sum_image / count
    variance = np.maximum(sum_squared / count - mean * mean, 0.0)
    return mean.astype(np.float64), np.sqrt(variance).astype(np.float64)


def adaptive_threshold(img, radius=15, method="bradley", t=0.15, k=0.34, R=128.0):
    """Apply Bradley-Roth or Sauvola thresholding; output contains 0 and 255."""
    image = _gray_uint8(img)
    if not np.isfinite(t) or not 0 <= t < 1:
        raise ValueError("t must be in [0, 1).")
    mean, std = local_mean_std(image, radius)
    if method.lower() in {"bradley", "bradley-roth", "bradley_roth"}:
        threshold = mean * (1.0 - t)
    elif method.lower() == "sauvola":
        if not np.isfinite(k) or not np.isfinite(R) or R <= 0:
            raise ValueError("Sauvola requires finite k and finite positive R.")
        threshold = mean * (1.0 + k * (std / R - 1.0))
    else:
        raise ValueError("method must be 'bradley' or 'sauvola'.")
    return np.where(image < threshold, 0, 255).astype(np.uint8)


def _gaussian_kernel(sigma):
    if sigma <= 0:
        return np.ones((1, 1), dtype=np.float64)
    radius = max(1, int(np.ceil(3 * sigma)))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    one_d = np.exp(-(axis * axis) / (2 * sigma * sigma))
    one_d /= one_d.sum()
    return np.outer(one_d, one_d)


def _convolve_hough(accumulator, kernel):
    """NumPy convolution with rho sign reversal across the theta pi seam."""
    rows, cols = accumulator.shape
    pr, pt = kernel.shape[0] // 2, kernel.shape[1] // 2
    row_kernel = kernel[pr]
    column_kernel = kernel[:, pt]
    if np.allclose(kernel, np.outer(column_kernel, row_kernel)):
        theta_padded = (
            np.concatenate(
                (accumulator[::-1, -pt:], accumulator, accumulator[::-1, :pt]),
                axis=1,
            )
            if pt
            else accumulator
        )
        horizontal = np.zeros_like(accumulator, dtype=np.float64)
        for offset, weight in enumerate(row_kernel):
            horizontal += weight * theta_padded[:, offset:offset + cols]

        rho_padded = np.pad(horizontal, ((pr, pr), (0, 0)), mode="constant")
        result = np.zeros_like(accumulator, dtype=np.float64)
        for offset, weight in enumerate(column_kernel):
            result += weight * rho_padded[offset:offset + rows]
        return result

    rho_padded = np.pad(accumulator, ((pr, pr), (0, 0)), mode="constant")
    theta_padded = (
        np.concatenate(
            (rho_padded[::-1, -pt:], rho_padded, rho_padded[::-1, :pt]), axis=1
        )
        if pt
        else rho_padded
    )
    result = np.zeros_like(accumulator, dtype=np.float64)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            result += kernel[i, j] * theta_padded[i:i + rows, j:j + cols]
    return result


def hough_accumulate(
    edges, theta_range, rho_range, n_theta, n_rho, grad_dir=None, delta_deg=None
):
    """Accumulate Hough votes; theta_range and rho_range endpoints are inclusive."""
    edge_map = np.asarray(edges)
    if edge_map.ndim != 2:
        raise ValueError("edges must be a two-dimensional image.")
    if n_theta < 1 or n_rho < 2:
        raise ValueError("n_theta must be >= 1 and n_rho must be >= 2.")
    theta_min, theta_max = map(float, theta_range)
    rho_min, rho_max = map(float, rho_range)
    if theta_max <= theta_min or rho_max <= rho_min:
        raise ValueError("theta_range and rho_range must be increasing.")
    if (grad_dir is None) != (delta_deg is None):
        raise ValueError("grad_dir and delta_deg must be supplied together.")
    if delta_deg is not None and (not np.isfinite(delta_deg) or delta_deg < 0):
        raise ValueError("delta_deg must be finite and non-negative.")

    ys, xs = np.nonzero(edge_map)
    thetas = np.linspace(theta_min, theta_max, int(n_theta), endpoint=True)
    rhos = np.linspace(rho_min, rho_max, int(n_rho), endpoint=True)
    accumulator = np.zeros((int(n_rho), int(n_theta)), dtype=np.int64)
    if not len(xs):
        return accumulator, thetas, rhos

    if grad_dir is not None:
        directions = np.asarray(grad_dir, dtype=np.float64)
        if directions.shape != edge_map.shape or not np.all(
            np.isfinite(directions[edge_map > 0])
        ):
            raise ValueError("grad_dir must match edges and be finite at edge pixels.")
        edge_angles = directions[ys, xs]
        delta = np.deg2rad(delta_deg)
    rho_step = (rho_max - rho_min) / (n_rho - 1)
    for theta_index, theta in enumerate(thetas):
        if grad_dir is None:
            selected = np.ones(len(xs), dtype=bool)
        else:
            angle_error = np.abs(
                (edge_angles - theta + np.pi / 2) % np.pi - np.pi / 2
            )
            selected = angle_error <= delta
        if not np.any(selected):
            continue
        rho_values = xs[selected] * np.cos(theta) + ys[selected] * np.sin(theta)
        rho_indices = np.rint((rho_values - rho_min) / rho_step).astype(np.int64)
        valid = (rho_indices >= 0) & (rho_indices < n_rho)
        accumulator[:, theta_index] = np.bincount(
            rho_indices[valid], minlength=n_rho
        )
    return accumulator, thetas, rhos


def find_peaks(
    A, n_peaks, nms_radius=3, smooth_sigma=1.0, thetas=None, rhos=None
):
    """Find 2D accumulator peaks after Gaussian smoothing and seam-aware NMS."""
    accumulator = np.asarray(A)
    if (
        accumulator.ndim != 2
        or n_peaks < 0
        or nms_radius < 0
        or smooth_sigma < 0
    ):
        raise ValueError("Invalid accumulator or find_peaks parameters.")
    nr, nt = accumulator.shape
    if thetas is None:
        thetas = np.arange(nt, dtype=np.float64)
    if rhos is None:
        rhos = np.arange(nr, dtype=np.float64)
    if len(thetas) != nt or len(rhos) != nr:
        raise ValueError("thetas/rhos sizes must match the accumulator.")
    if n_peaks == 0 or not np.any(accumulator):
        return []

    smoothed = _convolve_hough(
        accumulator.astype(np.float64), _gaussian_kernel(smooth_sigma)
    )
    radius = int(nms_radius)
    padded = np.pad(smoothed, ((radius, radius), (0, 0)), mode="constant")
    if radius:
        padded = np.concatenate(
            (padded[::-1, -radius:], padded, padded[::-1, :radius]), axis=1
        )
    center = padded[radius:radius + nr, radius:radius + nt]
    is_maximum = center > 0
    for dr in range(-radius, radius + 1):
        for dt in range(-radius, radius + 1):
            if dr == 0 and dt == 0:
                continue
            neighbor = padded[
                radius + dr:radius + dr + nr,
                radius + dt:radius + dt + nt,
            ]
            is_maximum &= center >= neighbor
    peak_rows, peak_cols = np.nonzero(is_maximum)
    order = np.argsort(smoothed[peak_rows, peak_cols])[::-1]
    suppressed = np.zeros_like(is_maximum)
    peaks = []
    for order_index in order:
        row = int(peak_rows[order_index])
        col = int(peak_cols[order_index])
        if suppressed[row, col]:
            continue
        peaks.append(
            (float(rhos[row]), float(thetas[col]), float(accumulator[row, col]))
        )
        if len(peaks) >= n_peaks:
            break
        for dt in range(-radius, radius + 1):
            raw_col = col + dt
            wraps = raw_col // nt
            target_col = raw_col % nt
            for dr in range(-radius, radius + 1):
                target_row = row + dr
                if wraps % 2:
                    target_row = nr - 1 - target_row
                if 0 <= target_row < nr:
                    suppressed[target_row, target_col] = True
    return peaks


def _normalize_hough_line(rho, theta):
    turns = int(np.floor(theta / np.pi))
    theta -= turns * np.pi
    if turns % 2:
        rho = -rho
    return float(rho), float(theta)


def hough_multiscale(
    edges,
    grad_dir,
    n_levels=2,
    refine_factor=8,
    n_peaks=10,
    coarse_theta_step_deg=1.0,
    coarse_rho_step=2.0,
    delta_deg=10.0,
):
    """Coarse-to-fine Hough: refine only the candidates from the coarse level."""
    edge_map = np.asarray(edges)
    if edge_map.ndim != 2 or np.asarray(grad_dir).shape != edge_map.shape:
        raise ValueError("edges and grad_dir must be 2D arrays of the same shape.")
    if n_levels < 1 or refine_factor < 2 or n_peaks < 1:
        raise ValueError("Require n_levels >= 1, refine_factor >= 2, n_peaks >= 1.")
    if coarse_theta_step_deg <= 0 or coarse_rho_step <= 0:
        raise ValueError("Hough resolutions must be positive.")

    height, width = edge_map.shape
    diagonal = float(np.hypot(height, width))
    n_theta = max(2, int(np.ceil(180.0 / coarse_theta_step_deg)))
    theta_step = np.pi / n_theta
    n_rho = max(2, int(np.ceil(2 * diagonal / coarse_rho_step)) + 1)
    coarse, thetas, rhos = hough_accumulate(
        edge_map,
        (0.0, np.pi - theta_step),
        (-diagonal, diagonal),
        n_theta,
        n_rho,
        grad_dir=grad_dir,
        delta_deg=delta_deg,
    )
    candidates = find_peaks(
        coarse, n_peaks, nms_radius=3, smooth_sigma=1.0,
        thetas=thetas, rhos=rhos,
    )
    rho_step = float(rhos[1] - rhos[0])
    if not candidates:
        return []

    refined = []
    for rho_center, theta_center, votes in candidates:
        current_rho_step = rho_step
        current_theta_step = theta_step
        for _ in range(1, n_levels):
            count = 2 * int(refine_factor) + 1
            local, local_thetas, local_rhos = hough_accumulate(
                edge_map,
                (
                    theta_center - current_theta_step,
                    theta_center + current_theta_step,
                ),
                (
                    rho_center - current_rho_step,
                    rho_center + current_rho_step,
                ),
                count,
                count,
                grad_dir=grad_dir,
                delta_deg=delta_deg,
            )
            rho_center, theta_center, votes = find_peaks(
                local, 1, nms_radius=0, smooth_sigma=0.0,
                thetas=local_thetas, rhos=local_rhos,
            )[0]
            current_rho_step /= refine_factor
            current_theta_step /= refine_factor
        rho_center, theta_center = _normalize_hough_line(rho_center, theta_center)
        refined.append((rho_center, theta_center, votes))

    refined.sort(key=lambda line: line[2], reverse=True)
    unique = []
    for rho, theta, votes in refined:
        duplicate = False
        for old_rho, old_theta, _ in unique:
            angle_delta = abs(theta - old_theta)
            rho_compare = rho
            if angle_delta > np.pi / 2:
                angle_delta = np.pi - angle_delta
                rho_compare = -rho
            if (
                angle_delta < np.deg2rad(coarse_theta_step_deg) / 2
                and abs(rho_compare - old_rho) < coarse_rho_step / 2
            ):
                duplicate = True
                break
        if not duplicate:
            unique.append((rho, theta, votes))
    return [(rho, theta) for rho, theta, _ in unique[:n_peaks]]
