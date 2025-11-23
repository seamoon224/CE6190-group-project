from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace as nspace

import torch
from PIL import Image
from diffusers import LMSDiscreteScheduler

from diff_edit.model.constants import TORCH_SEED
from diff_edit.model.model_composer import ModelComposer


class DiffEditEditer:
    """Expose DiffEdit through the Semantic Image Translation Editer API."""

    def __init__(
        self,
        device: str = "cuda",
        num_samples: int = 10,
        seed: int = TORCH_SEED,
        vae_model: str = "stabilityai/sd-vae-ft-ema",
        tokenizer: str = "openai/clip-vit-large-patch14",
        text_encoder: str = "openai/clip-vit-large-patch14",
        unet: str = "CompVis/stable-diffusion-v1-4",
        inpainting: str = "runwayml/stable-diffusion-inpainting",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "scaled_linear",
        num_train_timesteps: int = 1000,
        return_blended_mask: bool = False,
    ):
        scheduler = LMSDiscreteScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            num_train_timesteps=num_train_timesteps,
        )

        self.diffedit = ModelComposer(
            vae_model,
            tokenizer,
            text_encoder,
            unet,
            inpainting,
            scheduler,
            torch_dev=device,
        ).compose()

        self.args = nspace(
            device=device,
            num_samples=num_samples,
            seed=seed,
            return_blended_mask=return_blended_mask,
        )

    def __call__(self, img: Image.Image, src: str, tgt: str):
        if img.mode != "RGB":
            img = img.convert("RGB")

        tmp_path = None
        try:
            with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name)
                tmp_path = tmp.name

            torch.manual_seed(self.args.seed)
            mask, _, blended = self.diffedit.create_mask(
                tmp_path, src, tgt, n=self.args.num_samples, seed=self.args.seed
            )
            edited = self.diffedit.inpaint_mask_with_prompt(
                tmp_path, mask, tgt, seed=self.args.seed
            )
        finally:
            if tmp_path:
                path = Path(tmp_path)
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        outputs = {"images": edited}
        if self.args.return_blended_mask:
            outputs["mask"] = blended
        return outputs
