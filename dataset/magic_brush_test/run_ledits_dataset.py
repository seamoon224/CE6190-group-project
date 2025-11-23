"""
运行指令示例：
Full-set:
nohup python run_ledits_dataset.py \
  --resume \
  --images_json images.json \
  --images_root images \
  --output_root output_ledits \
  --caption_path global_descriptions.json \
  --metric 'clip-i,clip-t,lpips' \
  --eval_every 50 \
  --device cuda > ledits_dataset_log.txt 2>&1 &

Sub-set:
nohup python run_ledits_dataset.py \
  --resume \
  --images_json subset/images.json \
  --images_root subset/images \
  --output_root subset/output_ledits/L-7 \
  --caption_path subset/global_descriptions.json \
  --metric 'clip-i,clip-t,lpips' \
  --eval_every 50 \
  --device cuda > subset/output_ledits/L-7/ledits_dataset_log.txt 2>&1 &
"""

import os
import io
import json
import time
import argparse
from typing import Union, List, Tuple

import numpy as np
import torch
import PIL
import requests
from PIL import Image
from tqdm import tqdm

from scipy import spatial
import clip
import lpips
from torch import nn
from torchvision.transforms import transforms

from leditspp.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from leditspp import StableDiffusionPipeline_LEDITS


# ============================ LEDITS 工具函数 ============================

def load_image(image: Union[str, PIL.Image.Image]):
    """支持本地路径 / URL / PIL.Image"""
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def setup_ledits(device: str = "cuda"):
    """加载 LEDITS++ pipeline，一次就好"""
    model = "runwayml/stable-diffusion-v1-5"  # 或者你的本地路径
    print(f"Loading LEDITS++ pipeline from: {model}")

    pipe = StableDiffusionPipeline_LEDITS.from_pretrained(model, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(
        model, subfolder="scheduler", algorithm_type="sde-dpmsolver++", solver_order=2
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
    # 1. 读入原图（数据集里的 input），这里统一 resize 到 512x512
    org = load_image(input_path).resize((512, 512))
    im = np.array(org)[:, :, :3]

    # 2. 固定随机种子，保证可复现
    gen = torch.Generator(device=device).manual_seed(seed)

    # 3. Inversion（构造初始噪声轨迹）
    with torch.no_grad():
        _ = pipe.invert(
            im,
            num_inversion_steps=num_inversion_steps,
            generator=gen,
            verbose=False,
            skip=skip,
        )

        # 4. 编辑阶段：这里只用 edit_instruction
        out = pipe(
            editing_prompt=[edit_instruction],  # 单个 edit_prompt
            edit_threshold=[0.7],               # 可以之后调参
            edit_guidance_scale=[4.0],
            reverse_editing_direction=[False],  # False = 往 prompt 描述的方向加
            use_intersect_mask=True,
            # use_intersect_mask=False,
        )

    return out.images[0]  # PIL.Image


# ============================ Eval 函数（基于 image_eval.py） ============================

def eval_lpips_pairs(
    device: torch.device,
    image_pairs: List[Tuple[str, str]],
    lpips_fn,
):
    """
    Calculate LPIPS distance between generated and GT images.
    Lower is better (0 means identical in LPIPS feature space).
    """
    to_lpips_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    eval_score = 0.0
    for gen_path, gt_path in tqdm(image_pairs, desc="LPIPS (batch)", leave=False):
        gen_img = Image.open(gen_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        # resize to the same size
        gen_img = gen_img.resize(gt_img.size)

        gen_tensor = to_lpips_tensor(gen_img).unsqueeze(0).to(device)
        gt_tensor = to_lpips_tensor(gt_img).unsqueeze(0).to(device)

        with torch.no_grad():
            dist = lpips_fn(gen_tensor, gt_tensor).item()
        eval_score += dist

    return eval_score / len(image_pairs)


def eval_distance_pairs(
    image_pairs: List[Tuple[str, str]],
    metric: str = "l1",
):
    """
    Using pytorch to evaluate l1 or l2 distance
    """
    if metric == "l1":
        criterion = nn.L1Loss()
    elif metric == "l2":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    eval_score = 0.0
    for gen_path, gt_path in tqdm(image_pairs, desc=f"{metric.upper()} (batch)", leave=False):
        gen_img = Image.open(gen_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
        # resize to gt size
        gen_img = gen_img.resize(gt_img.size)
        # convert to tensor
        gen_img = transforms.ToTensor()(gen_img)
        gt_img = transforms.ToTensor()(gt_img)
        # calculate distance
        per_score = criterion(gen_img, gt_img).detach().cpu().numpy().item()
        eval_score += per_score

    return eval_score / len(image_pairs)


def eval_clip_i_pairs(
    device: torch.device,
    image_pairs: List[Tuple[str, str]],
    model,
    transform,
    metric: str = "clip_i",
):
    """
    Calculate CLIP-I / DINO score, the cosine similarity between the generated image and the ground truth image
    """
    def encode(image, model, transform):
        image_input = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            if metric == "clip_i":
                image_features = model.encode_image(image_input).detach().cpu().float()
            elif metric == "dino":
                image_features = model(image_input).detach().cpu().float()
            else:
                raise ValueError(f"Unknown metric type in eval_clip_i: {metric}")
        return image_features

    eval_score = 0.0
    for gen_path, gt_path in tqdm(image_pairs, desc=f"{metric.upper()} (batch)", leave=False):
        gen_img = Image.open(gen_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        gen_feat = encode(gen_img, model, transform)
        gt_feat = encode(gt_img, model, transform)

        similarity = 1 - spatial.distance.cosine(
            gen_feat.view(gen_feat.shape[1]),
            gt_feat.view(gt_feat.shape[1]),
        )
        if similarity > 1 or similarity < -1:
            raise ValueError("Strange similarity value")
        eval_score += similarity

    return eval_score / len(image_pairs)


def eval_clip_t_pairs(
    device: torch.device,
    image_pairs: List[Tuple[str, str]],
    model,
    transform,
    caption_dict: dict,
):
    """
    Calculate CLIP-T score, the cosine similarity between image features and text CLIP embedding
    Returns:
        gen_clip_t (float): average sim(gen_img, caption)
        gt_clip_t  (float): average sim(gt_img, caption)  -- oracle
    """
    def encode_img(image, model, transform):
        image_input = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input).detach().cpu().float()
        return image_features

    gen_clip_t = 0.0
    gt_clip_t = 0.0

    for gen_path, gt_path in tqdm(image_pairs, desc="CLIP-T (batch)", leave=False):
        gen_img = Image.open(gen_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        img_id = os.path.basename(os.path.dirname(gt_path))  # e.g. "139"
        gt_name = os.path.basename(gt_path)                  # e.g. "139-output1.png"

        if img_id not in caption_dict or gt_name not in caption_dict[img_id]:
            raise KeyError(f"Caption not found for img_id={img_id}, filename={gt_name}")

        gt_caption = caption_dict[img_id][gt_name]

        gen_feat = encode_img(gen_img, model, transform)
        gt_feat = encode_img(gt_img, model, transform)

        # get text CLIP embedding
        text_tokens = clip.tokenize(gt_caption).to(device)
        with torch.no_grad():
            text_feat = model.encode_text(text_tokens).detach().cpu().float()

        gen_sim = 1 - spatial.distance.cosine(
            gen_feat.view(gen_feat.shape[1]),
            text_feat.view(text_feat.shape[1]),
        )
        gt_sim = 1 - spatial.distance.cosine(
            gt_feat.view(gt_feat.shape[1]),
            text_feat.view(text_feat.shape[1]),
        )

        gen_clip_t += gen_sim
        gt_clip_t += gt_sim

    n = len(image_pairs)
    return gen_clip_t / n, gt_clip_t / n


def evaluate_batch(
    device: torch.device,
    batch_pairs: List[Tuple[str, str]],
    metrics: List[str],
    caption_dict: dict,
    clip_model=None,
    clip_preprocess=None,
    lpips_fn=None,
):
    """
    对当前 batch 的 (gen_path, gt_path) 做一次 evaluation，只算这批。
    返回：一个 dict，例如
    {
        "l1": ...,
        "clip-i": ...,
        "clip-t": ...,
        "clip-t_oracle": ...,
        "lpips": ...
    }
    """
    results = {}

    if "l1" in metrics:
        score = eval_distance_pairs(batch_pairs, metric="l1")
        print(f"[Batch Eval] L1: {score}")
        results["l1"] = score

    if "l2" in metrics:
        score = eval_distance_pairs(batch_pairs, metric="l2")
        print(f"[Batch Eval] L2: {score}")
        results["l2"] = score

    if "clip-i" in metrics:
        score = eval_clip_i_pairs(device, batch_pairs, clip_model, clip_preprocess, metric="clip_i")
        print(f"[Batch Eval] CLIP-I: {score}")
        results["clip-i"] = score

    if "clip-t" in metrics:
        gen_score, gt_score = eval_clip_t_pairs(device, batch_pairs, clip_model, clip_preprocess, caption_dict)
        print(f"[Batch Eval] CLIP-T: {gen_score}")
        print(f"[Batch Eval] CLIP-T Oracle: {gt_score}")
        results["clip-t"] = gen_score
        results["clip-t_oracle"] = gt_score

    if "lpips" in metrics:
        score = eval_lpips_pairs(device, batch_pairs, lpips_fn)
        print(f"[Batch Eval] LPIPS: {score}")
        results["lpips"] = score

    return results

# 统计 output_root 已经生成了多少张 output*.png
def count_existing_outputs(output_root):
    count = 0
    for img_id in os.listdir(output_root):
        img_dir = os.path.join(output_root, img_id)
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if "output" in fname and fname.endswith(".png"):
                count += 1
    return count

# ============================ Main ============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_json", type=str, required=True,
                        help="Path to images.json (描述 input / prompt / target 的文件)")
    parser.add_argument("--images_root", type=str, required=True,
                        help="GT 图像根目录，例如: /path/to/images")
    parser.add_argument("--output_root", type=str, required=True,
                        help="生成图像的保存根目录，例如: /path/to/output")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda 或 cpu")
    parser.add_argument("--num_inversion_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # eval 相关参数
    parser.add_argument("--caption_path", type=str, required=True,
                        help="global_description.json，格式: {img_id: {filename: caption}}")
    parser.add_argument("--metric", type=str, default="clip-i,clip-t,lpips",
                        help="Metrics to calculate (l1, l2, clip-i, clip-t, lpips)")
    parser.add_argument("--eval_every", type=int, default=50,
                        help="每多少张生成一次 evaluation（只评这 batch）")
    parser.add_argument("--eval_save_path", type=str, default="chunk_eval_results",
                        help="分批 evaluation 结果保存目录")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing outputs")

    args = parser.parse_args()

    already_done = 0
    if args.resume:
        already_done = count_existing_outputs(args.output_root)
        print(f"[Resume] Found {already_done} generated outputs. Will skip these.")

    device_str = args.device
    device = torch.device(device_str if device_str is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 解析 metric 列表
    metrics = [m.strip() for m in args.metric.split(",") if m.strip() != ""]

    # 载入 images.json
    with open(args.images_json, "r") as f:
        data = json.load(f)

    # 统计总共多少个 turn（就是总编辑次数）
    total_turns = sum(len(v) for v in data.values())
    print(f"Total edits to run: {total_turns}")

    # 载入 caption（给 CLIP-T 用）
    with open(args.caption_path, "r") as f:
        caption_dict = json.load(f)

    # 载入 CLIP / LPIPS（只在需要的时候加载）
    clip_model = None
    clip_preprocess = None
    lpips_fn = None

    if ("clip-i" in metrics) or ("clip-t" in metrics):
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        print("CLIP model loaded.")

    if "lpips" in metrics:
        lpips_fn = lpips.LPIPS(net="alex").to(device)
        lpips_fn.eval()
        print("LPIPS model (alex) loaded.")

    # ---------- 推理 + 时间统计 ----------
    start_time = time.time()

    # 初始化 LEDITS++
    pipe = setup_ledits(device=device)

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.eval_save_path, exist_ok=True)

    pbar = tqdm(total=total_turns, desc="Running LEDITS++ on dataset")

    processed = 0  # 全局已完成的 turn 数
    current_pairs: List[Tuple[str, str]] = []  # 当前 batch 的 (gen_path, gt_path)
    all_batch_results = []  # 记录每一批的评价指标
    global_index = 0  # 记录全局 index

    # 按 img_id -> 按 turn 顺序遍历
    for img_id, edit_list in data.items():
        # 每个 img_id 一个子目录
        save_dir = os.path.join(args.output_root, img_id)
        os.makedirs(save_dir, exist_ok=True)

        for turn_idx, item in enumerate(edit_list, start=1):
            if global_index < already_done:
                global_index += 1
                processed += 1
                pbar.update(1)
                continue  # 跳过已生成的图像
            input_name = item["input"]           # 如 "242679-input.png" / "368667-output1.png"
            original_prompt = item["original_prompt"]   # 暂时没用到
            edit_instruction = item["edit_instruction"]
            target_name = item["target"]         # 如 "242679-output1.png"

            print(f"input_name: {input_name}, edit_instruction: {edit_instruction}, target_name: {target_name}")

            input_path = os.path.join(args.images_root, img_id, input_name)
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input image not found: {input_path}")

            edit_prompt = edit_instruction  # 你以后也可以改成 original_prompt + edit_instruction

            # 为了每个 turn 种子不同，但又可复现，可以基于 img_id + turn_idx 构造
            try:
                base_seed = int(img_id)
            except ValueError:
                base_seed = args.seed
            cur_seed = base_seed + turn_idx

            edited_img = run_edit_single_example(
                pipe=pipe,
                device=device_str,
                input_path=input_path,
                edit_instruction=edit_prompt,
                num_inversion_steps=args.num_inversion_steps,
                seed=cur_seed,
                skip=0.15,
            )

            # 保存：结构必须和 eval_single_turn.py 预期一致
            save_path = os.path.join(save_dir, target_name)  # e.g. output/368667/368667-output1.png
            edited_img.save(save_path)

            # 和 GT 配对，留给 batch eval 用
            gt_path = os.path.join(args.images_root, img_id, target_name)
            current_pairs.append((save_path, gt_path))

            processed += 1
            pbar.set_postfix({"img_id": img_id, "turn": turn_idx})
            pbar.update(1)

            # 时间统计：已用时间 & 预计剩余时间
            elapsed = time.time() - start_time
            avg_per_job = elapsed / max(processed, 1)
            remaining = avg_per_job * (total_turns - already_done - processed)

            tqdm.write(
                f"[{processed}/{total_turns}] "
                f"elapsed: {elapsed/60:.1f} min, "
                f"ETA: {remaining/60:.1f} min"
            )

            # 每 eval_every 张做一次 evaluation（只评这一 batch）
            if len(current_pairs) == args.eval_every:
                print(f"\n=== Batch evaluation for samples {processed - len(current_pairs) + 1} ~ {processed} ===")
                batch_result = evaluate_batch(
                    device=device,
                    batch_pairs=current_pairs,
                    metrics=metrics,
                    caption_dict=caption_dict,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess,
                    lpips_fn=lpips_fn,
                )
                batch_result["start_idx"] = processed - len(current_pairs) + 1
                batch_result["end_idx"] = processed
                all_batch_results.append(batch_result)
                # 清空当前 batch
                current_pairs = []
            global_index += 1

    # 结束后，如果还有剩余不足 eval_every 的样本，也评一下
    if len(current_pairs) > 0:
        start_idx = total_turns - len(current_pairs) + 1
        end_idx = total_turns
        print(f"\n=== Final batch evaluation for samples {start_idx} ~ {end_idx} ===")
        batch_result = evaluate_batch(
            device=device,
            batch_pairs=current_pairs,
            metrics=metrics,
            caption_dict=caption_dict,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            lpips_fn=lpips_fn,
        )
        batch_result["start_idx"] = start_idx
        batch_result["end_idx"] = end_idx
        all_batch_results.append(batch_result)

    pbar.close()
    total_elapsed = time.time() - start_time
    print("All inference done.")
    print(f"Total elapsed time: {total_elapsed/60:.2f} minutes")
    print("All images processed. Generated outputs are saved to:", args.output_root)

    # 保存分批 evaluation 的结果
    save_json_path = os.path.join(args.eval_save_path, "chunk_eval_metrics.json")
    with open(save_json_path, "w") as f:
        json.dump(all_batch_results, f, indent=4)
    print(f"Per-batch evaluation metrics saved to: {save_json_path}")