import sys
import types
from pathlib import Path

import torch
from torchao.quantization import float8_weight_only, int8_weight_only, quantize_

sys.path.extend([str(Path(__file__).parent), str(Path(__file__).parent.parent)])

from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE

from _flux_forward_orig import forward_orig
from _utils import has_affordable_memory, is_newer_than_ada_lovelace

torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")


class FluxAccelerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "do_compile": ("BOOLEAN", {"default": True}),
                "mmdit_skip_blocks": ("STRING", {"default": "3,6,8,12"}),
                "dit_skip_blocks": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL", "VAE")
    FUNCTION = "acclerate"
    CATEGORY = "advanced/model"

    def __init__(self):
        self._compiled = False
        self._quantized = False

    def acclerate(
        self,
        model: ModelPatcher,
        vae: VAE,
        do_compile: bool,
        mmdit_skip_blocks: str,
        dit_skip_blocks: str,
    ) -> tuple[ModelPatcher, VAE]:
        diffusion_model = model.model.diffusion_model
        ae = vae.first_stage_model

        if not self._quantized:
            if ae.parameters().__next__().dtype in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
                torch.float8_e4m3fnuz,
                torch.float8_e5m2fnuz,
                torch.int8,
            ):
                pass
            elif is_newer_than_ada_lovelace(torch.device(0)):
                quantize_(ae, float8_weight_only())
            else:
                quantize_(ae, int8_weight_only())

            self._quantized = True

        if do_compile and not self._compiled:
            compile_mode = (
                "reduce-overhead"
                if has_affordable_memory(torch.device(0))
                else "default"
            )

            diffusion_model = diffusion_model.to(memory_format=torch.channels_last)
            diffusion_model = torch.compile(
                diffusion_model,
                mode=compile_mode,
                fullgraph=True,
            )

            ae = ae.to(memory_format=torch.channels_last)
            ae = torch.compile(
                ae,
                mode=compile_mode,
                fullgraph=True,
            )

            self.compiled = True

        model.model.diffusion_model = diffusion_model
        vae.first_stage_model = ae

        model.model.diffusion_model.mmdit_skip_blocks_ = [
            int(x) for x in mmdit_skip_blocks.split(",") if x
        ]
        model.model.diffusion_model.dit_skip_blocks_ = [
            int(x) for x in dit_skip_blocks.split(",") if x
        ]

        diffusion_model.forward_orig = types.MethodType(forward_orig, diffusion_model)

        return (model, vae)


NODE_CLASS_MAPPINGS = {"🍭FluxAccelerator": FluxAccelerator}
