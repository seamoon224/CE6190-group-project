"""
nohup python run_ledits_semantic_translation.py \
   --csv_path dataset/queries.csv \
   --img_root Imagenet_eval_padded \
   --output_root ledits_semantic_output \
   --device cuda \
   --resume > ledits_semantic_translation.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python run_ledits_semantic_translation.py \
--csv_path Imagenet_subset/queries.csv \
--img_root Imagenet_eval_padded \
--output_root Imagenet_subset/L-18 \
--device cuda \
--resume > Imagenet_subset/ledits_semantic_translation.log 2>&1 &
"""

import os
import csv
import time
import argparse
from typing import Union

import numpy as np
import torch
import PIL
import requests
from PIL import Image
from tqdm import tqdm

from leditspp.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from leditspp import StableDiffusionPipeline_LEDITS


# ============================ 通用工具 ============================

def load_image(image: Union[str, PIL.Image.Image]):
    """支持本地路径 / URL / PIL.Image，统一转成 RGB。"""
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, "
                f"and {image} is not a valid path"
            )
    elif isinstance(image, PIL.Image.Image):
        pass
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, "
            "a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def setup_ledits(device: str = "cuda"):
    """加载 LEDITS++ pipeline，一次就好。"""
    model = "runwayml/stable-diffusion-v1-5"  # 如果你有本地模型，也可以改成本地路径
    print(f"Loading LEDITS++ pipeline from: {model}")

    pipe = StableDiffusionPipeline_LEDITS.from_pretrained(model, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(
        model,
        subfolder="scheduler",
        algorithm_type="sde-dpmsolver++",
        solver_order=2,
    )
    pipe.to(device)
    return pipe


def run_edit_single_example(
    pipe,
    device: str,
    input_path: str,
    edit_instruction: str,
    num_inversion_steps: int = 50,
    seed: int = 42,
    skip: float = 0.15,
):
    """
    对单张图片进行 inversion + edit，返回 PIL.Image
    """
    # org = load_image(input_path).resize((512, 512))
    # 读取已经 padding 好的 512x512
    org = load_image(input_path)
    assert org.size == (512, 512), f"Expected 512x512 input, got {org.size}"
    im = np.array(org)[:, :, :3]

    gen = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        _ = pipe.invert(
            im,
            num_inversion_steps=num_inversion_steps,
            generator=gen,
            verbose=False,
            skip=skip,
        )

        out = pipe(
            editing_prompt=[edit_instruction],
            edit_threshold=[0.7],
            edit_guidance_scale=[8.0],
            reverse_editing_direction=[False],
            use_intersect_mask=True,
        )

    return out.images[0]


# ============================ 主逻辑：跑 Semantic Image Translation ============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="queries.csv 的路径（包含 i,path,source,target,is_test 等列）",
    )
    parser.add_argument(
        "--img_root",
        type=str,
        required=True,
        help="ImageNet val 的根目录（使得 os.path.join(img_root, path) 能找到图片）",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="生成图像保存目录，文件名为 {i}.jpeg",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda 或 cpu",
    )
    parser.add_argument(
        "--num_inversion_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="如果已存在 {i}.jpeg 就跳过（断点续跑）",
    )

    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    device_str = args.device
    device = torch.device(
        device_str if device_str is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 1. 读取 CSV，只保留 is_test=True 的行
    rows = []
    with open(args.csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            flag = str(row.get("is_test", "")).strip().lower()
            if flag == "true":
                rows.append(row)

    print(f"Found {len(rows)} rows with is_test == True.")

    # 2. 初始化 LEDITS++
    pipe = setup_ledits(device=device_str)

    start_time = time.time()
    processed = 0

    for row in tqdm(rows, desc="Running LEDITS++ on SemanticImageTranslation"):
        idx = row["i"]          # e.g. "3741"
        rel_path = row["path"]  # e.g. "n02091032/ILSVRC2012_val_00042240.JPEG"
        src = row["source"]     # e.g. "Italian greyhound"
        tgt = row["target"]     # e.g. "Afghan hound"

        # in_path = os.path.join(args.img_root, rel_path)
        # 只保留文件名
        filename = os.path.basename(rel_path)
        in_path = os.path.join(args.img_root, filename)
        out_path = os.path.join(args.output_root, f"{idx}.jpeg")

        if args.resume and os.path.exists(out_path):
            # 已经生成过，跳过
            processed += 1
            continue

        if not os.path.exists(in_path):
            raise FileNotFoundError(f"Input image not found: {in_path}")

        # 你可以按需改这个 prompt，这里给一个相对自然的版本
        edit_instruction = f"Turn the {src} into a {tgt}."

        # 为了可复现，可以用 i 作为 seed 偏移
        try:
            base = int(idx)
        except ValueError:
            base = args.seed
        cur_seed = args.seed + base

        edited_img = run_edit_single_example(
            pipe=pipe,
            device=device_str,
            input_path=in_path,
            edit_instruction=edit_instruction,
            num_inversion_steps=args.num_inversion_steps,
            seed=cur_seed,
            skip=0.15,
        )

        edited_img.save(out_path, format="JPEG")
        processed += 1

        elapsed = time.time() - start_time
        avg = elapsed / max(processed, 1)
        remaining = avg * (len(rows) - processed)
        tqdm.write(
            f"[{processed}/{len(rows)}] "
            f"elapsed: {elapsed/60:.1f} min, "
            f"ETA: {remaining/60:.1f} min"
        )

    total_elapsed = time.time() - start_time
    print("All inference done.")
    print(f"Total elapsed time: {total_elapsed/60:.2f} minutes")
    print("All images processed. Generated outputs are saved to:", args.output_root)


if __name__ == "__main__":
    main()