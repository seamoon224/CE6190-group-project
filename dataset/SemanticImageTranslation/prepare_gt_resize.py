"""
python prepare_gt_resize.py \
    --src_root Imagenet_eval \
    --dst_root Imagenet_eval_padded \
    --size 512
"""

import os
from PIL import Image
from tqdm import tqdm
import argparse

def resize_with_padding(img, target=512):
    """等比例缩放 + pad 到 target×target"""
    w, h = img.size
    scale = target / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_resized = img.resize((new_w, new_h), Image.BICUBIC)

    new_img = Image.new("RGB", (target, target), (0, 0, 0))
    offset_x = (target - new_w) // 2
    offset_y = (target - new_h) // 2
    new_img.paste(img_resized, (offset_x, offset_y))
    return new_img


def process_folder(src_root, dst_root, target=512):
    os.makedirs(dst_root, exist_ok=True)

    files = [
        f for f in os.listdir(src_root)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(files)} images to process.")

    for fname in tqdm(files):
        src_path = os.path.join(src_root, fname)
        dst_path = os.path.join(dst_root, fname)

        img = Image.open(src_path).convert("RGB")
        new_img = resize_with_padding(img, target=target)
        new_img.save(dst_path, format="JPEG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_root", required=True,
        help="原始 GT 图像目录（你的 Imagenet_eval）"
    )
    parser.add_argument(
        "--dst_root", required=True,
        help="输出目录（新的 padded GT 图像）"
    )
    parser.add_argument(
        "--size", type=int, default=512,
        help="最终输出大小（默认 512×512）"
    )
    args = parser.parse_args()

    process_folder(args.src_root, args.dst_root, args.size)


if __name__ == "__main__":
    main()
