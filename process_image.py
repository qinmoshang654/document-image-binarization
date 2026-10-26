import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.algorithms import Parameters, segment


def main():
    parser = argparse.ArgumentParser(description="将浅色背景、深色文字的文档转为二值图")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--method",
        choices=("otsu", "sauvola", "tuned_sauvola", "normalized", "denoised", "improved"),
        default="improved",
        help="细小文字可选denoised，跳过单像素连通域删除",
    )
    args = parser.parse_args()
    configuration = Path(__file__).parent / "results" / "selected_parameters.json"
    if configuration.exists():
        selected = json.loads(configuration.read_text(encoding="utf-8"))
        parameters = Parameters(**selected["parameters"])
    else:
        parameters = Parameters()
    image = np.array(Image.open(args.input).convert("L"))
    foreground = segment(image, args.method, parameters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 内部True表示文字，保存时映射为黑色，便于直接查看。
    Image.fromarray(np.where(foreground, 0, 255).astype(np.uint8)).save(args.output)
    print(f"已保存：{args.output}")


if __name__ == "__main__":
    main()
