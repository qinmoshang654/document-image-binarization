import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


CONDITIONS = ("uniform", "shadow", "noise", "mixed")
VALIDATION_SEEDS = range(100, 104)
TEST_SEEDS = range(300, 312)


def locate_font(font_path=None):
    """优先使用指定字体；Windows默认用宋体生成中文样本。"""
    candidates = [font_path] if font_path else []
    candidates += ["C:/Windows/Fonts/simsun.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise FileNotFoundError("Provide --font with a valid TrueType font path")


def make_clean(seed, font_path):
    """根据布局种子生成清洁文字页，不在这一步加入退化。"""
    random = np.random.default_rng(seed)
    image = Image.new("L", (800, 560), 255)
    draw = ImageDraw.Draw(image)
    font_size = int(random.integers(19, 25))
    font = ImageFont.truetype(font_path, font_size)
    chinese = Path(font_path).name.lower().startswith("simsun")
    topics = [
        "数字图像处理与分析", "灰度直方图与阈值分割", "空间滤波与图像增强",
        "图像中的目标和背景", "局部均值与标准差", "噪声抑制和细节保护",
    ]
    for row in range(11):
        left = int(random.integers(25, 55))
        top = 24 + row * 45
        number = int(random.integers(1000, 9999))
        if chinese:
            topic = topics[int(random.integers(0, len(topics)))]
            text = f"{row + 1:02d}  {topic}  {number}"
        else:
            text = f"{row + 1:02d}  Image processing experiment  {number}"
        draw.text((left, top), text, font=font, fill=0)
    return np.array(image, dtype=np.uint8)


def degrade(clean, seed, condition):
    """在清洁页上叠加指定退化，随机状态不与布局生成共用。"""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown degradation condition: {condition}")
    random = np.random.default_rng(seed)
    height, width = clean.shape
    vertical, horizontal = np.mgrid[0:height, 0:width]
    horizontal = horizontal / (width - 1)
    vertical = vertical / (height - 1)
    ink = (255.0 - clean) / 255.0
    contrast = float(random.uniform(115, 170))
    image = 243.0 - contrast * ink
    if condition in {"shadow", "mixed"}:
        # 横向渐变模拟光照不均，高斯形暗带模拟局部阴影。
        center = float(random.uniform(0.2, 0.8))
        illumination = 0.48 + 0.47 * horizontal + 0.04 * vertical
        illumination -= 0.25 * np.exp(-((horizontal - center) ** 2) / 0.018)
        image = image * illumination
    if condition in {"noise", "mixed"}:
        image += random.normal(0.0, 9.0, clean.shape)
        blurred = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image = np.array(blurred.filter(ImageFilter.GaussianBlur(0.35)), dtype=float)
        impulse = random.random(clean.shape)
        image[impulse < 0.006] = 0
        image[impulse > 0.994] = 255
    return np.clip(image, 0, 255).astype(np.uint8)


def generate(output, font_path):
    """按布局划分验证和测试数据，返回输入与真值的对应清单。"""
    output = Path(output)
    records = []
    for split, seeds in (("validation", VALIDATION_SEEDS), ("test", TEST_SEEDS)):
        for seed in seeds:
            clean = make_clean(seed, font_path)
            for condition_index, condition in enumerate(CONDITIONS):
                sample_id = f"{split}_{seed}_{condition}"
                image = degrade(clean, seed * 10 + condition_index, condition)
                directory = output / split
                directory.mkdir(parents=True, exist_ok=True)
                image_path = directory / f"{sample_id}.png"
                truth_path = directory / f"{sample_id}_gt.png"
                Image.fromarray(image).save(image_path)
                # 标注取自退化前的清洁页，不能从带噪输入反推“真值”。
                Image.fromarray(np.where(clean < 128, 0, 255).astype(np.uint8)).save(truth_path)
                records.append({
                    "id": sample_id, "split": split, "layout_seed": seed,
                    "noise_seed": seed * 10 + condition_index, "condition": condition,
                    "image": image_path.as_posix(), "truth": truth_path.as_posix(),
                    "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                })
    return records
