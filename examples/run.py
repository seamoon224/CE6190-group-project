# LEDITS++ example code
import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import requests
from typing import Union
import PIL
from leditspp.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from leditspp import  StableDiffusionPipeline_LEDITS

def load_image(image: Union[str, PIL.Image.Image]):
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

def image_grid(imgs, rows, cols, spacing = 20):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size

    grid = Image.new('RGBA', size=(cols * w + (cols-1)*spacing, rows * h + (rows-1)*spacing ), color=(255,255,255,0))

    for i, img in enumerate(imgs):
        grid.paste(img, box=( i // rows * (w+spacing), i % rows * (h+spacing)))
    return grid

model = 'runwayml/stable-diffusion-v1-5'
#model = '/workspace/StableDiff/models/stable-diffusion-v1-5'

device = 'cuda'

pipe = StableDiffusionPipeline_LEDITS.from_pretrained(model, safety_checker = None,)
pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(model, subfolder="scheduler", algorithm_type="sde-dpmsolver++", solver_order=2)
pipe.to(device)

# Let us now generate some of the examples presented in the paper

# org = load_image('https://github.com/ml-research/ledits_pp/blob/main/examples/images/yann-lecun.jpg?raw=true').resize((512,512))

org = load_image('/home/haiyue/codes/project/baseline/CE6190/algorithms/ledits_pp/dataset/magic_brush_test/images/242679/242679-input.png')
im = np.array(org)[:, :, :3]

gen = torch.manual_seed(42)
with torch.no_grad():
    _ = pipe.invert(im, num_inversion_steps=50, generator=gen, verbose=True, skip=0.15)
    # out = pipe(editing_prompt=['george clooney', 'sunglasses'],     # 这里需要改edit prompt，为什么可以有多个输入呢？
    out = pipe(editing_prompt=['Put a cat on the seat.'],     # 这里需要改edit prompt，为什么可以有多个输入呢？
               edit_threshold=[.7],
               edit_guidance_scale=[3],
               reverse_editing_direction=[False],
               use_intersect_mask=True,)
    
# Save result
save_path = "ledits_output.png"
out.images[0].save(save_path)
print(f"Saved edited image to: {save_path}")