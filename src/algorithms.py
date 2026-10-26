from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import median_filter, uniform_filter
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.morphology import remove_small_objects


@dataclass(frozen=True)
class Parameters:
    """改进流程的参数；默认值对应随附验证集选出的配置。"""

    window: int = 41
    k: float = 0.3
    background_window: int = 81
    min_area: int = 2

    def to_dict(self):
        return asdict(self)


def box_mean(image, window):
    """利用积分图计算方框均值，边界按镜像扩展。"""
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")
    radius = window // 2
    # 累加前转浮点，避免灰度值在整数类型中溢出。
    padded = np.pad(np.asarray(image, dtype=np.float64), radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    ) / (window * window)


def normalize_background(image, window):
    """估计缓变的纸张背景，再把各处亮度拉到相近的范围。"""
    background = box_mean(image, window)
    return np.clip(image / np.maximum(background, 1.0) * 230.0, 0, 255).astype(np.uint8)


def suppress_impulses(image):
    """只替换孤立的黑白极值，不对整幅图直接做中值滤波。"""
    median = median_filter(image, size=3, mode="reflect")
    dark_support = uniform_filter((image < 30).astype(float), size=3, mode="reflect") * 9
    light_support = uniform_filter((image > 250).astype(float), size=3, mode="reflect") * 9
    # 计数包含中心。小于1.5表示附近没有同类极值，也留出浮点误差余量。
    isolated = ((image < 30) & (dark_support < 1.5)) | ((image > 250) & (light_support < 1.5))
    outlier = np.abs(image.astype(float) - median.astype(float)) > 60
    return np.where(isolated & outlier, median, image).astype(np.uint8)


def segment(image, method="improved", parameters=None):
    """返回文字掩膜：True为深色文字，False为背景。"""
    image = np.asarray(image)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("input must be a 2-D uint8 grayscale image")
    parameters = parameters or Parameters()
    if method == "otsu":
        return image <= threshold_otsu(image)
    if method == "sauvola":
        return image <= threshold_sauvola(image, window_size=25, k=0.2, r=127.5)
    if method not in {"tuned_sauvola", "normalized", "denoised", "improved"}:
        raise ValueError(f"unknown method: {method}")
    working = image
    if method in {"denoised", "improved"}:
        working = suppress_impulses(working)
    if method != "tuned_sauvola":
        working = normalize_background(working, parameters.background_window)
    foreground = working <= threshold_sauvola(
        working, window_size=parameters.window, k=parameters.k, r=127.5
    )
    if method == "improved":
        foreground = remove_small_objects(foreground, min_size=parameters.min_area, connectivity=2)
    return foreground


def evaluate(prediction, truth):
    """按文字前景计算指标；分母为零的指标按0记录。"""
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have equal shapes")
    true_positive = int(np.count_nonzero(prediction & truth))
    false_positive = int(np.count_nonzero(prediction & ~truth))
    false_negative = int(np.count_nonzero(~prediction & truth))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1),
        "iou": true_positive / max(true_positive + false_positive + false_negative, 1),
        "pixel_error": float(np.mean(prediction != truth)),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }
