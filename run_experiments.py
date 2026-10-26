import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import data
from skimage.filters import threshold_sauvola

from src.algorithms import Parameters, evaluate, segment
from src.dataset import CONDITIONS, TEST_SEEDS, VALIDATION_SEEDS, generate, locate_font


ROOT = Path(__file__).resolve().parent
METHODS = ("otsu", "sauvola", "tuned_sauvola", "normalized", "denoised", "improved")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_sample(record):
    with Image.open(ROOT / record["image"]) as source:
        image = np.array(source.convert("L"))
    with Image.open(ROOT / record["truth"]) as source:
        truth = np.array(source.convert("L")) < 128
    if image.shape != truth.shape:
        raise ValueError(f"图像与标注尺寸不一致：{record['id']}")
    return image, truth


def validate_manifest(records):
    """复用数据前核对文件哈希，并确认两组没有共享文字布局。"""
    layouts = {"validation": set(), "test": set()}
    for record in records:
        layouts[record["split"]].add(record["layout_seed"])
        path = ROOT / record["image"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"输入文件与清单不符：{record['id']}")
    if not all(layouts.values()) or layouts["validation"] & layouts["test"]:
        raise ValueError("验证集和测试集必须非空，且不能共享布局")


def select_parameters(validation):
    """只用验证集选窗口和k，测试图不进入这个函数。"""
    search = []
    for window, sensitivity in itertools.product((25, 41), (0.2, 0.3, 0.4, 0.5)):
        parameters = Parameters(window=window, k=sensitivity)
        scores = [
            evaluate(segment(image, "improved", parameters), truth)["f1"]
            for image, truth in validation
        ]
        search.append({
            "parameters": parameters.to_dict(),
            "validation_f1": float(np.mean(scores)),
        })
    return max(search, key=lambda item: item["validation_f1"]), search


def summarize(measurements):
    """先算逐图指标，再按退化类型取均值，不合并不同图的像素。"""
    summary = []
    for condition, method in itertools.product(("all",) + CONDITIONS, METHODS):
        subset = [
            row for row in measurements
            if row["method"] == method
            and (condition == "all" or row["condition"] == condition)
        ]
        entry = {"condition": condition, "method": method, "n": len(subset)}
        for metric in ("precision", "recall", "f1", "iou", "pixel_error", "ms"):
            values = [row[metric] for row in subset]
            entry[metric] = float(np.mean(values))
            entry[metric + "_std"] = float(np.std(values, ddof=1))
        summary.append(entry)
    return summary


def comparison(images, labels, destination):
    """拼接输入、真值和预测，只用于展示，不参与指标计算。"""
    tile_width, tile_height = 400, 280
    columns = 2 if len(images) <= 4 else 3
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * (tile_height + 32)), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/times.ttf")
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    for index, (image, label) in enumerate(zip(images, labels)):
        tile = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
        tile.thumbnail((tile_width, tile_height))
        left = index % columns * tile_width
        top = index // columns * (tile_height + 32)
        draw.text((left + 8, top + 4), label, fill="black", font=font)
        canvas.paste(tile, (left, top + 32))
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description="重跑文档分割对比实验")
    parser.add_argument("--font", help="重新生成数据时使用的字体路径")
    parser.add_argument("--reuse-data", action="store_true", help="使用随附PNG，不重新生成文字页")
    args = parser.parse_args()
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    manifest_path = ROOT / "data" / "manifest.json"
    if args.reuse_data:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        font_path = locate_font(args.font)
        records = generate(ROOT / "data", font_path)
        for record in records:
            for key in ("image", "truth"):
                record[key] = Path(record[key]).relative_to(ROOT).as_posix()
        write_json(manifest_path, records)
        write_json(ROOT / "data" / "generation.json", {
            "font": Path(font_path).name,
            "font_sha256": hashlib.sha256(Path(font_path).read_bytes()).hexdigest(),
            "size": [800, 560],
            "gt_rule": "clean < 128",
            "validation_layouts": list(VALIDATION_SEEDS),
            "test_layouts": list(TEST_SEEDS),
        })
    validate_manifest(records)
    validation = [load_sample(record) for record in records if record["split"] == "validation"]
    selected, search = select_parameters(validation)
    parameters = Parameters(**selected["parameters"])
    write_json(results / "validation_search.json", search)
    write_json(results / "selected_parameters.json", selected)
    measurements = []
    for record in records:
        if record["split"] != "test":
            continue
        image, truth = load_sample(record)
        for method in METHODS:
            # 只计分割耗时，读写文件和计算评价指标均放在计时区间外。
            start = time.perf_counter()
            prediction = segment(image, method, parameters)
            elapsed = (time.perf_counter() - start) * 1000
            measurements.append({"id": record["id"], "condition": record["condition"], "method": method, **evaluate(prediction, truth), "ms": elapsed})
            output = results / "predictions" / method
            output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.where(prediction, 0, 255).astype(np.uint8)).save(output / f"{record['id']}.png")
    with (results / "per_image_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measurements[0]))
        writer.writeheader()
        writer.writerows(measurements)
    summary = summarize(measurements)
    write_json(results / "summary.json", summary)
    for condition in CONDITIONS:
        record = next(record for record in records if record["id"] == f"test_300_{condition}")
        image, truth = load_sample(record)
        images = [image, np.where(truth, 0, 255)] + [np.where(segment(image, method, parameters), 0, 255) for method in ("otsu", "sauvola", "denoised", "improved")]
        labels = ["Input", "Ground truth", "Otsu", "Sauvola (original)", "Denoised + normalized", "Improved"]
        comparison(images, labels, results / f"comparison_{condition}.png")
    # 扫描页没有像素真值，只出对比图，不混进上面的合成数据统计。
    real_page = data.page()
    Image.fromarray(real_page).save(ROOT / "data" / "real_page.png")
    comparison([real_page] + [np.where(segment(real_page, method, parameters), 0, 255) for method in ("otsu", "sauvola", "improved")], ["Real scanned page", "Otsu", "Sauvola (original)", "Improved"], results / "comparison_real.png")
    upstream = ROOT / "upstream"
    upstream.mkdir(exist_ok=True)
    source_path = Path(inspect.getfile(threshold_sauvola))
    (upstream / "thresholding_original.py").write_bytes(source_path.read_bytes())
    package = importlib.metadata.distribution("scikit-image")
    for file in package.files:
        if str(file).replace("\\", "/").endswith("dist-info/LICENSE.txt"):
            (upstream / "LICENSE.txt").write_bytes(Path(package.locate_file(file)).read_bytes())
    environment = {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(), "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-image", "Pillow")}, "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "test_images": 48, "validation_images": 16, "timing": "one sequential pass per image; excludes disk IO and metric evaluation; no warmup"}
    write_json(results / "environment.json", environment)
    print(json.dumps({"selected": selected, "overall": [row for row in summary if row["condition"] == "all"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
