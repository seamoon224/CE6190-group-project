"""
Single-turn image editing evaluation:
- Only evaluate ALL TURNS (no final_turn concept)
- Support metrics: L1, L2, CLIP-I, CLIP-T, LPIPS
- Naming style:
    images/
      139/
        139-input.png         # ignored
        139-output1.png       # GT turn 1
        139-output2.png       # GT turn 2
        139-output3.png       # GT turn 3
    output/
      139/
        139-output1.png       # GEN turn 1
        139-output2.png       # GEN turn 2
        139-output3.png       # GEN turn 3

python image_eval.py \
  --generated_path subset/output_ledits/L-7 \
  --gt_path subset/images \
  --caption_path subset/global_descriptions.json \
  --metric 'clip-i,clip-t,lpips' \
  --device cuda \
  --save_path results_eval_all
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import io
import json
import os
import time

import torch
import clip
import lpips
from PIL import Image
from scipy import spatial
from torch import nn
from torchvision.transforms import transforms
from tqdm import tqdm
from pyiqa.utils.img_util import is_image_file


########################### Basic Func ################################

def imread(img_source, rgb=False, target_size=None):
    """Read image
    Args:
        img_source (str, bytes, or PIL.Image): image filepath string, image contents as a bytearray or a PIL Image instance
        rgb: convert input to RGB if true
        target_size: resize image to target size if not None
    """
    if isinstance(img_source, bytes):
        img = Image.open(io.BytesIO(img_source))
    elif isinstance(img_source, str):
        assert is_image_file(img_source), f'{img_source} is not a valid image file.'
        img = Image.open(img_source)
    elif isinstance(img_source, Image.Image):
        img = img_source
    else:
        raise Exception("Unsupported source type")
    if rgb:
        img = img.convert('RGB')
    if target_size is not None:
        img = img.resize(target_size, Image.BICUBIC)
    return img


########################### Evaluation ################################

def eval_lpips(args, image_pairs, lpips_fn):
    """
    Calculate LPIPS distance between generated and GT images.
    Lower is better (0 means identical in LPIPS feature space).
    """
    to_lpips_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    eval_score = 0.0
    for gen_path, gt_path in tqdm(image_pairs, desc="LPIPS"):
        gen_img = Image.open(gen_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        # resize to the same size
        gen_img = gen_img.resize(gt_img.size)

        gen_tensor = to_lpips_tensor(gen_img).unsqueeze(0).to(args.device)
        gt_tensor = to_lpips_tensor(gt_img).unsqueeze(0).to(args.device)

        with torch.no_grad():
            dist = lpips_fn(gen_tensor, gt_tensor).item()
        eval_score += dist

    return eval_score / len(image_pairs)


def eval_distance(image_pairs, metric='l1'):
    """
    Using pytorch to evaluate l1 or l2 distance
    """
    if metric == 'l1':
        criterion = nn.L1Loss()
    elif metric == 'l2':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    eval_score = 0.0
    for gen_path, gt_path in tqdm(image_pairs, desc=f"{metric.upper()}"):
        gen_img = Image.open(gen_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        # resize to gt size
        gen_img = gen_img.resize(gt_img.size)
        # convert to tensor
        gen_img = transforms.ToTensor()(gen_img)
        gt_img = transforms.ToTensor()(gt_img)
        # calculate distance
        per_score = criterion(gen_img, gt_img).detach().cpu().numpy().item()
        eval_score += per_score

    return eval_score / len(image_pairs)


def eval_clip_i(args, image_pairs, model, transform, metric='clip_i'):
    """
    Calculate CLIP-I / DINO score, the cosine similarity between the generated image and the ground truth image
    """
    def encode(image, model, transform):
        image_input = transform(image).unsqueeze(0).to(args.device)
        with torch.no_grad():
            if metric == 'clip_i':
                image_features = model.encode_image(image_input).detach().cpu().float()
            elif metric == 'dino':
                image_features = model(image_input).detach().cpu().float()
            else:
                raise ValueError(f"Unknown metric type in eval_clip_i: {metric}")
        return image_features

    eval_score = 0.0
    for gen_path, gt_path in tqdm(image_pairs, desc=f"{metric.upper()}"):
        gen_img = Image.open(gen_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        gen_feat = encode(gen_img, model, transform)
        gt_feat = encode(gt_img, model, transform)

        similarity = 1 - spatial.distance.cosine(
            gen_feat.view(gen_feat.shape[1]),
            gt_feat.view(gt_feat.shape[1])
        )
        if similarity > 1 or similarity < -1:
            raise ValueError("Strange similarity value")
        eval_score += similarity
        
    return eval_score / len(image_pairs)


def eval_clip_t(args, image_pairs, model, transform, caption_dict):
    """
    Calculate CLIP-T score, the cosine similarity between image features and text CLIP embedding
    Returns:
        gen_clip_t (float): average sim(gen_img, caption)
        gt_clip_t  (float): average sim(gt_img, caption)  -- oracle
    """
    def encode_img(image, model, transform):
        image_input = transform(image).unsqueeze(0).to(args.device)
        with torch.no_grad():
            image_features = model.encode_image(image_input).detach().cpu().float()
        return image_features

    gen_clip_t = 0.0
    gt_clip_t = 0.0
    
    for gen_path, gt_path in tqdm(image_pairs, desc="CLIP-T"):
        gen_img = Image.open(gen_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        img_id = os.path.basename(os.path.dirname(gt_path))   # e.g. "139"
        gt_name = os.path.basename(gt_path)                   # e.g. "139-output1.png"

        if img_id not in caption_dict or gt_name not in caption_dict[img_id]:
            raise KeyError(f"Caption not found for img_id={img_id}, filename={gt_name}")

        gt_caption = caption_dict[img_id][gt_name]

        gen_feat = encode_img(gen_img, model, transform)
        gt_feat = encode_img(gt_img, model, transform)

        # get text CLIP embedding
        text_tokens = clip.tokenize(gt_caption).to(args.device)
        with torch.no_grad():
            text_feat = model.encode_text(text_tokens).detach().cpu().float()

        gen_sim = 1 - spatial.distance.cosine(
            gen_feat.view(gen_feat.shape[1]),
            text_feat.view(text_feat.shape[1])
        )
        gt_sim = 1 - spatial.distance.cosine(
            gt_feat.view(gt_feat.shape[1]),
            text_feat.view(text_feat.shape[1])
        )

        gen_clip_t += gen_sim
        gt_clip_t += gt_sim
        
    n = len(image_pairs)
    return gen_clip_t / n, gt_clip_t / n


########################### Data Loading（single-turn） ################################

def load_data_single_turn(args):
    """
    Single-turn setting:
    For each img_id folder, pair
        GEN: all files containing 'output'
        GT : all files containing 'output'
    sorted by the numeric part in filename.

    E.g.
        generated_path/139/139-output1.png, 139-output2.png, 139-output3.png
        gt_path/139/139-output1.png, 139-output2.png, 139-output3.png
    """
    gen_ids = sorted([
        d for d in os.listdir(args.generated_path)
        if os.path.isdir(os.path.join(args.generated_path, d))
    ])
    gt_ids = sorted([
        d for d in os.listdir(args.gt_path)
        if os.path.isdir(os.path.join(args.gt_path, d))
    ])

    if set(gen_ids) != set(gt_ids):
        print("The directory names under generated path and gt path are not same!")
        print("In generated only:", set(gen_ids) - set(gt_ids))
        print("In gt only:", set(gt_ids) - set(gen_ids))
        raise ValueError("The directory names under generated path and gt path are not same.")

    all_turn_pairs = []

    for img_id in gen_ids:
        gen_dir = os.path.join(args.generated_path, img_id)
        gt_dir = os.path.join(args.gt_path, img_id)

        gen_names = [
            s for s in os.listdir(gen_dir)
            if (s.endswith('.png') or s.endswith('.jpg')) and ('mask' not in s)
        ]
        gt_names = [
            s for s in os.listdir(gt_dir)
            if (s.endswith('.png') or s.endswith('.jpg')) and ('mask' not in s)
        ]

        # only keep 'output' images (ignore input images like 139-input.png)
        gen_outputs = [s for s in gen_names if 'output' in s]
        gt_outputs = [s for s in gt_names if 'output' in s]

        if len(gen_outputs) == 0 or len(gt_outputs) == 0:
            raise ValueError(f"No output images found for img_id={img_id}")

        # sort by numeric part, e.g. output1 / output2 / output3
        gen_outputs.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or -1))
        gt_outputs.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or -1))

        if len(gen_outputs) != len(gt_outputs):
            print(f"WARNING: number of GEN and GT outputs not equal for img_id={img_id}")
            print("GEN:", gen_outputs)
            print("GT :", gt_outputs)
            raise ValueError(f"The number of turns in generated and gt images are not same for {img_id}")

        for g_name, t_name in zip(gen_outputs, gt_outputs):
            all_turn_pairs.append(
                (os.path.join(gen_dir, g_name),
                 os.path.join(gt_dir, t_name))
            )

    return all_turn_pairs


########################### Main ################################

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_workers',
                        type=int,
                        help=('Number of processes to use for data loading. '
                              'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='Device to use. Like cuda or cpu')
    parser.add_argument('--generated_path',
                        type=str,
                        help='Path of generated images (folders)')
    parser.add_argument('--gt_path',
                        type=str,
                        help='Path of gt images (folders)')
    parser.add_argument('--caption_path',
                        type=str,
                        default="global_description.json",
                        help='File path of captions for CLIP-T, '
                             'expected format: {img_id: {filename: caption}}')
    parser.add_argument('--metric',
                        type=str,
                        default='clip-i,clip-t,lpips',
                        help='Metrics to calculate (l1, l2, clip-i, clip-t, lpips)')
    parser.add_argument('--save_path',
                        type=str,
                        default='results',
                        help='Path to save the results')

    args = parser.parse_args()
    args.metric = [m.strip() for m in args.metric.split(',') if m.strip() != ""]

    for arg in vars(args):
        print(arg, getattr(args, arg))

    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    # load pairs (single-turn, all turns only)
    all_turn_pairs = load_data_single_turn(args)

    with open(args.caption_path, 'r') as f:
        caption_dict = json.load(f)

    print(f"No. of all turn pairs: {len(all_turn_pairs)}")
    print('#' * 50, 'ALL TURNS (first 10)', '#' * 50)
    for i in range(min(10, len(all_turn_pairs))):
        print(f"Pair {i}: GEN={all_turn_pairs[i][0]}  |  GT={all_turn_pairs[i][1]}")
    print('#' * 50, 'END ALL TURNS PREVIEW', '#' * 50)

    evaluated_metrics_dict = {'all_turn': {}}

    # Pixel distance metrics (optional)
    if 'l1' in args.metric:
        all_turn_eval_score = eval_distance(all_turn_pairs, 'l1')
        print(f"All turn L1 distance: {all_turn_eval_score}")
        evaluated_metrics_dict['all_turn']['l1'] = all_turn_eval_score

    if 'l2' in args.metric:
        all_turn_eval_score = eval_distance(all_turn_pairs, 'l2')
        print(f"All turn L2 distance: {all_turn_eval_score}")
        evaluated_metrics_dict['all_turn']['l2'] = all_turn_eval_score

    # CLIP-I
    if 'clip-i' in args.metric:
        model, transform = clip.load("ViT-B/32", args.device)
        print("CLIP model loaded for CLIP-I.")
        all_turn_eval_score = eval_clip_i(args, all_turn_pairs, model, transform, metric='clip_i')
        print(f"All turn CLIP-I: {all_turn_eval_score}")
        evaluated_metrics_dict['all_turn']['clip-i'] = all_turn_eval_score

    # CLIP-T
    if 'clip-t' in args.metric:
        model, transform = clip.load("ViT-B/32", args.device)
        print("CLIP model loaded for CLIP-T.")
        all_turn_eval_score, all_turn_oracle_score = eval_clip_t(args, all_turn_pairs, model, transform, caption_dict)
        print(f"All turn CLIP-T: {all_turn_eval_score}")
        print(f"All turn CLIP-T Oracle: {all_turn_oracle_score}")
        evaluated_metrics_dict['all_turn']['clip-t'] = all_turn_eval_score
        evaluated_metrics_dict['all_turn']['clip-t_oracle'] = all_turn_oracle_score

    # LPIPS
    if 'lpips' in args.metric:
        lpips_fn = lpips.LPIPS(net='alex').to(args.device)
        lpips_fn.eval()
        print("LPIPS model (alex) loaded.")
        all_turn_eval_score = eval_lpips(args, all_turn_pairs, lpips_fn)
        print(f"All turn LPIPS: {all_turn_eval_score}")
        evaluated_metrics_dict['all_turn']['lpips'] = all_turn_eval_score

    # pretty print
    print(evaluated_metrics_dict)
    print(f"Setting: all_turn")
    metrics = evaluated_metrics_dict['all_turn'].keys()
    print(f"{'Metric':<15}", end='|')
    for metric in metrics:
        print(f"{metric:<15}", end='|')
    print()
    print('-' * 16 * max(1, len(list(metrics))))
    print(f"{'Score':<15}", end='|')
    for metric in metrics:
        print(f"{evaluated_metrics_dict['all_turn'][metric]:<15.4f}", end='|')
    print()
    print('#' * 16 * max(1, len(list(metrics))))

    # save results
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, 'evaluation_metrics.json'), 'w') as f:
        json.dump(evaluated_metrics_dict, f, indent=4)
