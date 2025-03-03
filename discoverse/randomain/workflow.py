import random
from typing import List

import PIL.Image as pil_image
import numpy as np
import torch
from params_proto import ParamsProto
import cv2

from discoverse.randomain.utils import (
    get_value_at_index,
    import_custom_nodes,
    add_extra_model_paths,
)


def disable_comfy_args():
    """
    Hacky injection to disable comfy args parsing.
    """
    import comfy.options

    def enable_args_parsing(enable=False):
        global args_parsing
        args_parsing = enable

    comfy.options.enable_args_parsing = enable_args_parsing


def image_grid(img_list: List[List[pil_image.Image]]):
    rows = len(img_list)
    cols = len(img_list[0])

    w, h = img_list[0][0].size
    grid = pil_image.new("RGB", size=(cols * w, rows * h))

    for i, row in enumerate(img_list):
        for j, img in enumerate(row):
            grid.paste(img, box=(j * w, i * h))
    return grid


class ImageGen(ParamsProto, prefix="imagen", cli=False):
    """
    Image Generation from three semantic masks.

    foreground_prompt: str
    background_text: str
    cone_prompt: str (although this can be replaced with any third object)

    negative_text: str

    control_parameters:
        strength: float
        grow_mask_amount: int
        fore_grow_mask_amount: int

        background_strength: float

    """

    width = 1080
    height = 1080
    batch_size: int = 1

    num_steps = 7
    denoising_strength = 1.0

    control_strength = 0.8

    grow_mask_amount = 0
    fore_grow_mask_amount = 0

    background_strength = 0.5
    fore_strength = 1.5

    checkpoint_path = "sd_xl_turbo_1.0_fp16.safetensors"
    control_path = "controlnet_depth_sdxl_1.0.safetensors"
    vae_path = "sdxl_vae.safetensors"
    device = "cuda"

    def __post_init__(self):
        disable_comfy_args()
        add_extra_model_paths()
        import_custom_nodes()

        from nodes import (
            EmptyLatentImage,
            CheckpointLoaderSimple,
            NODE_CLASS_MAPPINGS,
            VAEDecode,
            CLIPTextEncode,
            ControlNetLoader,
        )

        checkpointloadersimple = CheckpointLoaderSimple()
        self.checkpoint = checkpointloadersimple.load_checkpoint(
            ckpt_name=self.checkpoint_path
        )
        self.clip_text_encode = CLIPTextEncode()
        self.empty_latent = EmptyLatentImage()

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        self.ksampler = ksamplerselect.get_sampler(sampler_name="lcm")

        controlnetloader = ControlNetLoader()
        self.controlnet = controlnetloader.load_controlnet(
            control_net_name=self.control_path
        )

        self.imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        self.growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
        self.vaedecode = VAEDecode()

        print("loading is done.")

    # @torch.no_grad
    # @staticmethod
    # def to_tensor(img: pil_image.Image):
    #     np_img = np.asarray(img)
    #     np_img = np.copy(np_img)
    #     return torch.Tensor(np_img) / 255.0

    def generate(
        self,
        _deps=None,
        *,
        depth,
        masks,
        prompt,
        **deps,
    ):

        from nodes import (
            ConditioningSetMask,
            ConditioningCombine,
            NODE_CLASS_MAPPINGS,
            ControlNetApply,
        )

        # we reference the class to take advantage of the namespacing
        ImageGen._update(_deps, **deps)

        fore_objs = [obj for obj in list(masks.keys()) if obj != "background"]

        # transfer to torch
        masks_t = {}
        for k, mask in masks.items():
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask_t = (torch.Tensor(mask) / 255.0)[None, ..., None].repeat([1, 1, 1, 3])
            masks_t[k] = mask_t
        depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
        depth_t = (torch.Tensor(depth) / 255.0)[None, ..., None].repeat([1, 1, 1, 3])

        with torch.inference_mode():
            emptylatentimage = self.empty_latent.generate(
                width=ImageGen.width,
                height=ImageGen.height,
                batch_size=ImageGen.batch_size,
            )

            textencodes = {}
            for obj in fore_objs + ["background", "negative"]:
                textencodes[obj] = self.clip_text_encode.encode(
                    text=prompt[obj], clip=get_value_at_index(self.checkpoint, 1)
                )

            conditioningsetmask = ConditioningSetMask()
            conditioningcombine = ConditioningCombine()
            controlnetapply = ControlNetApply()
            sdturboscheduler = NODE_CLASS_MAPPINGS["SDTurboScheduler"]()
            samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()

            # condition
            conditions = {}
            for obj in fore_objs + ["background"]:

                expand = (
                    ImageGen.grow_mask_amount
                    if obj == "background"
                    else ImageGen.fore_grow_mask_amount
                )
                image2mask = self.imagetomask.image_to_mask(
                    channel="red", image=get_value_at_index([masks_t[obj]], 0)
                )
                growmask = self.growmask.expand_mask(
                    expand=expand,
                    tapered_corners=True,
                    mask=get_value_at_index(image2mask, 0),
                )

                strength = (
                    ImageGen.background_strength
                    if obj == "background"
                    else ImageGen.fore_strength
                )
                conditions[obj] = conditioningsetmask.append(
                    strength=strength,
                    set_cond_area="default",
                    conditioning=get_value_at_index(textencodes[obj], 0),
                    mask=get_value_at_index(growmask, 0),
                )

            # cobination of conditions
            conditions_list = list(conditions.values())
            final_combine = conditioningcombine.combine(
                conditioning_1=get_value_at_index(conditions_list[0], 0),
                conditioning_2=get_value_at_index(conditions_list[1], 0),
            )
            for condition in conditions_list[2:]:
                final_combine = conditioningcombine.combine(
                    conditioning_1=get_value_at_index(final_combine, 0),
                    conditioning_2=get_value_at_index(condition, 0),
                )

            # control
            controlnetapply = controlnetapply.apply_controlnet(
                strength=ImageGen.control_strength,
                conditioning=get_value_at_index(final_combine, 0),
                control_net=get_value_at_index(self.controlnet, 0),
                image=get_value_at_index((depth_t,), 0),
            )

            sdturboscheduler = sdturboscheduler.get_sigmas(
                steps=ImageGen.num_steps,
                denoise=ImageGen.denoising_strength,
                model=get_value_at_index(self.checkpoint, 0),
            )

            samplercustom = samplercustom.sample(
                add_noise=True,
                noise_seed=random.randint(1, 2**64),
                cfg=1,
                model=get_value_at_index(self.checkpoint, 0),
                positive=get_value_at_index(controlnetapply, 0),
                negative=get_value_at_index(textencodes["negative"], 0),
                sampler=get_value_at_index(self.ksampler, 0),
                sigmas=get_value_at_index(sdturboscheduler, 0),
                latent_image=get_value_at_index(emptylatentimage, 0),
            )

            (image_batch,) = self.vaedecode.decode(
                samples=get_value_at_index(samplercustom, 0),
                vae=get_value_at_index(self.checkpoint, 2),
            )[:1]

            (generated_image,) = image_batch

            gen_np = (generated_image * 255).cpu().numpy().astype("uint8")
            gen_np = cv2.cvtColor(gen_np, cv2.COLOR_RGB2BGR)
            return gen_np
