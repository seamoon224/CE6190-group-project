"""
python run.py \
  --input ../Imagenet_eval/ILSVRC2012_val_00000023.JPEG \
  --out_dir resize_vis_00023
"""
import os
from PIL import Image, ImageOps

# === 三种预处理方法 ===

def resize_stretch(img, size=512):
    # 直接拉伸
    return img.resize((size, size), Image.BICUBIC)

def resize_shorter_side_and_center_crop(img, size=512):
    w, h = img.size
    if w < h:
        new_w = size
        new_h = int(h * size / w)
    else:
        new_h = size
        new_w = int(w * size / h)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    right = left + size
    bottom = top + size
    img = img.crop((left, top, right, bottom))
    return img

def resize_longer_side_and_pad(img, size=512, fill=0):
    w, h = img.size
    if w > h:
        new_w = size
        new_h = int(h * size / w)
    else:
        new_h = size
        new_w = int(w * size / h)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    delta_w = size - new_w
    delta_h = size - new_h
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )
    img = ImageOps.expand(img, padding, fill=fill)
    return img

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="原始图片路径")
    parser.add_argument("--out_dir", type=str, default="resize_debug", help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img = Image.open(args.input).convert("RGB")

    # 三种输出单张
    stretch = resize_stretch(img)
    crop = resize_shorter_side_and_center_crop(img)
    pad = resize_longer_side_and_pad(img)

    stretch.save(os.path.join(args.out_dir, "stretch_512.png"))
    crop.save(os.path.join(args.out_dir, "shorter_center_crop_512.png"))
    pad.save(os.path.join(args.out_dir, "longer_pad_512.png"))

    # 拼成一张 3×1 对比图（可选）
    W, H = 512, 512
    canvas = Image.new("RGB", (W * 3, H))
    canvas.paste(stretch, (0, 0))
    canvas.paste(crop, (W, 0))
    canvas.paste(pad, (W * 2, 0))
    canvas.save(os.path.join(args.out_dir, "compare_3x1.png"))

    print("Saved to:", args.out_dir)
