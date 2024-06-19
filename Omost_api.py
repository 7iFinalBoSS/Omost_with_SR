import os
import argparse
import re
import cv2

import lib_omost.memory_management as memory_management
import uuid
import torch
import numpy as np
import gradio as gr
import tempfile

# Phi3 Hijack
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel
Phi3PreTrainedModel._supports_sdpa = True

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from lib_omost.pipeline import StableDiffusionXLOmostPipeline
from chat_interface import ChatInterface
from transformers.generation.stopping_criteria import StoppingCriteriaList
import lib_omost.canvas as omost_canvas

# super resolution module
from extension.Real_ESRGAN.realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

class OmostAutoPipeline():
    def __init__(self, 
                sdxl_model_name: str = '/path/to/RealVisXL_V4.0',
                llm_model_name: str = '/path/to/omost-llama-3-8b-4bits',
                sr_model_path: str = '/path/to/RealESRGAN_x4plus.pth', # depend on sr_model_name
                num_samples: int = 1, 
                seed: int = 1337, 
                image_width: int = 1024, 
                image_height: int = 1024, 
                highres_scale: float = 2.0, 
                steps: int = 30, 
                cfg: float = 5.0, 
                highres_steps: int = 20,
                highres_denoise: float = 0.4,
                negative_prompt: str = 'nsfw,lowres,bad anatomy,bad hands,text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry',
                sr_model_name: str = 'RealESRGAN_x4plus', # RealESRGAN_x4plus_anime_6B
                tile: int = 0,
                tile_pad: int = 10,
                pre_pad: int = 0,
                fp32: bool = True,
    ):
        # image generation default settings
        self.num_samples = num_samples
        self.seed = seed
        self.image_width = image_width
        self.image_height = image_height
        self.highres_scale = highres_scale
        self.steps = steps
        self.cfg = cfg
        self.highres_steps = highres_steps
        self.highres_denoise = highres_denoise
        self.negative_prompt = negative_prompt

        ################################################# SDXL #################################################
        self.tokenizer = CLIPTokenizer.from_pretrained(sdxl_model_name, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(sdxl_model_name, subfolder="tokenizer_2")
        self.text_encoder = CLIPTextModel.from_pretrained(sdxl_model_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
        self.text_encoder_2 = CLIPTextModel.from_pretrained(sdxl_model_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
        self.vae = AutoencoderKL.from_pretrained(sdxl_model_name, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16") # bfloat16 vae
        self.unet = UNet2DConditionModel.from_pretrained(sdxl_model_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16")

        self.unet.set_attn_processor(AttnProcessor2_0())
        self.vae.set_attn_processor(AttnProcessor2_0())

        self.pipeline = StableDiffusionXLOmostPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            text_encoder_2=self.text_encoder_2,
            tokenizer_2=self.tokenizer_2,
            unet=self.unet,
            scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
        )
        memory_management.unload_all_models([self.text_encoder, self.text_encoder_2, self.vae, self.unet])

        ################################################# LLM #################################################
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.bfloat16,  # This is computation type, not load/memory type. The loading quant type is baked in config.
            device_map="auto"  # This will load model to gpu with an offload system
        )

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name
        )
        memory_management.unload_all_models(self.llm_model)

        ################################################# SR #################################################
        # RealESRGAN module
        if sr_model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            sr_netscale = 4
        else:  # RealESRGAN_x4plus_anime_6B  # x4 RRDBNet model with 6 blocks
            sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            sr_netscale = 4
        
        self.upsampler = RealESRGANer(
            scale=sr_netscale,
            model_path=sr_model_path,
            dni_weight=None,
            model=sr_model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=not fp32,
            gpu_id=None)

    def set_num_samples(self, num_samples: int):
        self.num_samples = num_samples

    def set_seed(self, seed: int):
        self.seed = seed

    def set_image_width(self, image_width: int):
        self.image_width = image_width

    def set_image_height(self, image_height: int):
        self.image_height = image_height

    def set_highres_scale(self, highres_scale: float):
        self.highres_scale = highres_scale

    def set_steps(self, steps: int):
        self.steps = steps

    def set_cfg(self, cfg: float):
        self.cfg = cfg

    def set_highres_steps(self, highres_steps: int):
        self.highres_steps = highres_steps
    
    def set_highres_denoise(self, highres_denoise: float):
        self.highres_denoise = highres_denoise
    
    def set_negative_prompt(self, negative_prompt: str):
        self.negative_prompt = negative_prompt
    
    def PIL_to_cv2(self, numpy_array):
        if numpy_array.ndim == 2:  # grey image
            cv2_array = numpy_array
        elif numpy_array.shape[2] == 3:  # RGB image
            cv2_array = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
        elif numpy_array.shape[2] == 4:  # RGBA image
            cv2_array = cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2BGRA)
        else:
            raise ValueError("Unsupported image format")
        return cv2_array

    def cv2_to_PIL(self, numpy_array):
        if numpy_array.ndim == 2: # grey image
            pil_array = numpy_array
        elif numpy_array.shape[2] == 3:  # BGR image
            pil_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
        elif numpy_array.shape[2] == 4:  # BGRA image
            pil_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGRA2RGBA)
        else:
            raise ValueError("Unsupported image format")
        return pil_array 

    def img_sr_api(self, numpy_array):
        img = self.PIL_to_cv2(numpy_array)
        try:
            # outscale is upsampling scale of the image
            output, _ = self.upsampler.enhance(img, outscale=2)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            return numpy_array # return origin input if error
        return self.cv2_to_PIL(output)

    @torch.inference_mode()
    def pytorch2numpy(self, imgs):
        results = []
        for x in imgs:
            y = x.movedim(0, -1)
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
            results.append(y)
        return results

    @torch.inference_mode()
    def numpy2pytorch(self, imgs):
        h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
        h = h.movedim(-1, 1)
        return h

    def resize_without_crop(self, image, target_width, target_height):
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
        return np.array(resized_image)
    
    @torch.inference_mode()
    def llm_output(self, message: str, temperature: float=0.6, top_p: float=0.9, max_new_tokens: int=4096, repetition_penalty: float=1.2) -> str:
        np.random.seed(int(self.seed))
        torch.manual_seed(int(self.seed))

        conversation = [{"role": "system", "content": omost_canvas.system_prompt}]
        conversation.append({"role": "user", "content": message})
        
        memory_management.load_models_to_gpu(self.llm_model)

        input_ids = self.llm_tokenizer.apply_chat_template(
            conversation, return_tensors="pt", add_generation_prompt=True).to(self.llm_model.device)

        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty # Prevents LLM from looping indefinitely, leading to subsequent failure to match
        )

        if temperature == 0:
            generate_kwargs['do_sample'] = False

        # call llm_model.generate method
        output_token_ids = self.llm_model.generate(**generate_kwargs)
        outputs = self.llm_tokenizer.decode(output_token_ids[0][len(input_ids[0]):], skip_prompt=True, skip_special_tokens=True)

        return outputs

    @torch.inference_mode()
    def post_process(self, response, message):
        def process_canvas(response):
            try:
                canvas = omost_canvas.Canvas.from_bot_response(response)
                canvas_outputs = canvas.process()
                return canvas_outputs
            except Exception as e:
                print('The assistant response is not valid canvas:', e)
                return None

        canvas_outputs = process_canvas(response)
        
        if canvas_outputs and canvas_outputs.get("bag_of_conditions") is not None:
            return canvas_outputs
        else:
            print("The canvas outputs failed to parse, re-parsing!")
            self.seed += 1
            llm_outputs = self.llm_output(message)
            return self.post_process(llm_outputs, message)

    @torch.inference_mode()
    def diffusion_output(self, canvas_outputs, num_samples, seed, image_width, image_height, highres_scale, steps, cfg, highres_steps, highres_denoise, negative_prompt):
        use_initial_latent = False
        eps = 0.05

        image_width, image_height = int(image_width // 64) * 64, int(image_height // 64) * 64
        rng = torch.Generator(device=memory_management.gpu).manual_seed(seed)
        memory_management.load_models_to_gpu([self.text_encoder, self.text_encoder_2])
        positive_cond, positive_pooler, negative_cond, negative_pooler = self.pipeline.all_conds_from_canvas(canvas_outputs, negative_prompt)

        if use_initial_latent:
            memory_management.load_models_to_gpu([self.vae])
            initial_latent = torch.from_numpy(canvas_outputs['initial_latent'])[None].movedim(-1, 1) / 127.5 - 1.0
            initial_latent_blur = 40
            initial_latent = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(initial_latent, (initial_latent_blur,) * 4, mode='reflect'),
                kernel_size=(initial_latent_blur * 2 + 1,) * 2, stride=(1, 1))
            initial_latent = torch.nn.functional.interpolate(initial_latent, (image_height, image_width))
            initial_latent = initial_latent.to(dtype=self.vae.dtype, device=self.vae.device)
            initial_latent = self.vae.encode(initial_latent).latent_dist.mode() * self.vae.config.scaling_factor
        else:
            initial_latent = torch.zeros(size=(num_samples, 4, image_height // 8, image_width // 8), dtype=torch.float32)

        memory_management.load_models_to_gpu([self.unet])
        initial_latent = initial_latent.to(dtype=self.unet.dtype, device=self.unet.device)

        latents = self.pipeline(
            initial_latent=initial_latent,
            strength=1.0,
            num_inference_steps=int(steps),
            batch_size=num_samples,
            prompt_embeds=positive_cond,
            negative_prompt_embeds=negative_cond,
            pooled_prompt_embeds=positive_pooler,
            negative_pooled_prompt_embeds=negative_pooler,
            generator=rng,
            guidance_scale=float(cfg),
        ).images

        memory_management.load_models_to_gpu([self.vae])
        latents = latents.to(dtype=self.vae.dtype, device=self.vae.device) / self.vae.config.scaling_factor
        pixels = self.vae.decode(latents).sample
        B, C, H, W = pixels.shape
        pixels = self.pytorch2numpy(pixels)

        if highres_scale > 1.0 + eps:
            # add super resolution module
            pixels = [self.img_sr_api(p) for p in pixels]
            pixels = [
                self.resize_without_crop(
                    image=p,
                    target_width=int(round(W * highres_scale / 64.0) * 64),
                    target_height=int(round(H * highres_scale / 64.0) * 64)
                ) for p in pixels
            ]

            pixels = self.numpy2pytorch(pixels).to(device=self.vae.device, dtype=self.vae.dtype)
            latents = self.vae.encode(pixels).latent_dist.mode() * self.vae.config.scaling_factor

            memory_management.load_models_to_gpu([self.unet])
            latents = latents.to(device=self.unet.device, dtype=self.unet.dtype)

            latents = self.pipeline(
                initial_latent=latents,
                strength=highres_denoise,
                num_inference_steps=highres_steps,
                batch_size=num_samples,
                prompt_embeds=positive_cond,
                negative_prompt_embeds=negative_cond,
                pooled_prompt_embeds=positive_pooler,
                negative_pooled_prompt_embeds=negative_pooler,
                generator=rng,
                guidance_scale=float(cfg),
            ).images

            memory_management.load_models_to_gpu([self.vae])
            latents = latents.to(dtype=self.vae.dtype, device=self.vae.device) / self.vae.config.scaling_factor
            pixels = self.vae.decode(latents).sample
            pixels = self.pytorch2numpy(pixels)
        
        # return PIL images list
        return [Image.fromarray(p) for p in pixels]
    
    # call omost to generate images
    # the process is: txt input -> llm -> canvus outputs -> txt2img -> sr -> img2img -> img output
    def omost_generate(self, message: str, num_samples: int=None, seed: int=None, image_width: int=None, image_height: int=None) -> list:
        # Using llm to generate canvus code for adding global and local descriptions
        response = self.llm_output(message)
        # Processing the response to generate canvus outputs
        canvas_outputs = self.post_process(response, message)
        # Image render
        output = self.diffusion_output(canvas_outputs, 
                                    num_samples = num_samples if num_samples is not None else self.num_samples, 
                                    seed = seed if seed is not None else self.seed, 
                                    image_width = image_width if image_width is not None else self.image_width, 
                                    image_height = image_height if image_height is not None else self.image_height, 
                                    highres_scale = self.highres_scale, 
                                    steps = self.steps, 
                                    cfg = self.cfg, 
                                    highres_steps = self.highres_steps,
                                    highres_denoise = self.highres_denoise, 
                                    negative_prompt = self.negative_prompt)

        return output

if __name__ == "__main__":
    # a simple test for OmostAutoPipeline
    pipe = OmostAutoPipeline()
    topic = "the beautiful city"
    result = pipe.omost_generate(topic)
    result[0].save("result.png")