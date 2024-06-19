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

def PIL_to_cv2(numpy_array):
    if numpy_array.ndim == 2:  # grey image
        cv2_array = numpy_array
    elif numpy_array.shape[2] == 3:  # RGB image
        cv2_array = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    elif numpy_array.shape[2] == 4:  # RGBA image
        cv2_array = cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2BGRA)
    else:
        raise ValueError("Unsupported image format")
    return cv2_array

def cv2_to_PIL(numpy_array):
    if numpy_array.ndim == 2: # grey image
        pil_array = numpy_array
    elif numpy_array.shape[2] == 3:  # BGR image
        pil_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
    elif numpy_array.shape[2] == 4:  # BGRA image
        pil_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError("Unsupported image format")
    return pil_array 

def img_sr_api(numpy_array, upsampler):
    img = PIL_to_cv2(numpy_array)
    try:
        # outscale is upsampling scale of the image
        output, _ = upsampler.enhance(img, outscale=args.highres_scale)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        return numpy_array # return origin input if error
    return cv2_to_PIL(output)


def sanitize_filename(filename, replacement_char='_'):
    illegal_chars = r'<>:"/\|?*'
    sanitized_filename = re.sub(f'[{re.escape(illegal_chars)}]', replacement_char, filename.replace(" ",replacement_char))
    sanitized_filename = sanitized_filename.rstrip('. ')
    if not sanitized_filename:
        sanitized_filename = 'untitled'
    return sanitized_filename

@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results

@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h

def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

@torch.inference_mode()
def llm_output(message: str, seed:int=12345, temperature: float=0.6, top_p: float=0.9, max_new_tokens: int=4096) -> str:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    # Replacement of dialog forms with single input
    conversation = [{"role": "system", "content": omost_canvas.system_prompt}]
    conversation.append({"role": "user", "content": message})
    
    memory_management.load_models_to_gpu(llm_model)

    input_ids = llm_tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True).to(llm_model.device)

    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.2 # Prevents LLM from looping indefinitely, leading to subsequent failure to match
    )

    if temperature == 0:
        generate_kwargs['do_sample'] = False

    # call llm_model.generate method
    output_token_ids = llm_model.generate(**generate_kwargs)
    outputs = llm_tokenizer.decode(output_token_ids[0][len(input_ids[0]):], skip_prompt=True, skip_special_tokens=True)

    return outputs

@torch.inference_mode()
def post_process(response, message, seed):
    def process_canvas(response, message, seed):
        try:
            canvas = omost_canvas.Canvas.from_bot_response(response)
            canvas_outputs = canvas.process()
            return canvas_outputs
        except Exception as e:
            print('The assistant response is not valid canvas:', e)
            return None

    canvas_outputs = process_canvas(response, message, seed)
    
    if canvas_outputs and canvas_outputs.get("bag_of_conditions") is not None:
        return canvas_outputs
    else:
        print("The canvas outputs failed to parse, re-parsing!")
        new_seed = seed + 1
        llm_outputs = llm_output(message, new_seed)
        return post_process(llm_outputs, message, new_seed)

@torch.inference_mode()
def diffusion_output(canvas_outputs, num_samples, seed, image_width, image_height, highres_scale, steps, cfg, highres_steps, highres_denoise, negative_prompt) -> list:
    use_initial_latent = False
    eps = 0.05

    image_width, image_height = int(image_width // 64) * 64, int(image_height // 64) * 64
    rng = torch.Generator(device=memory_management.gpu).manual_seed(seed)
    memory_management.load_models_to_gpu([text_encoder, text_encoder_2])
    positive_cond, positive_pooler, negative_cond, negative_pooler = pipeline.all_conds_from_canvas(canvas_outputs, negative_prompt)

    if use_initial_latent:
        memory_management.load_models_to_gpu([vae])
        initial_latent = torch.from_numpy(canvas_outputs['initial_latent'])[None].movedim(-1, 1) / 127.5 - 1.0
        initial_latent_blur = 40
        initial_latent = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(initial_latent, (initial_latent_blur,) * 4, mode='reflect'),
            kernel_size=(initial_latent_blur * 2 + 1,) * 2, stride=(1, 1))
        initial_latent = torch.nn.functional.interpolate(initial_latent, (image_height, image_width))
        initial_latent = initial_latent.to(dtype=vae.dtype, device=vae.device)
        initial_latent = vae.encode(initial_latent).latent_dist.mode() * vae.config.scaling_factor
    else:
        initial_latent = torch.zeros(size=(num_samples, 4, image_height // 8, image_width // 8), dtype=torch.float32)

    memory_management.load_models_to_gpu([unet])
    initial_latent = initial_latent.to(dtype=unet.dtype, device=unet.device)

    latents = pipeline(
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

    memory_management.load_models_to_gpu([vae])
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    pixels = vae.decode(latents).sample
    B, C, H, W = pixels.shape
    pixels = pytorch2numpy(pixels)

    if highres_scale > 1.0 + eps:
        # add super resolution module
        pixels = [img_sr_api(p,upsampler) for p in pixels]
        pixels = [
            resize_without_crop(
                image=p,
                target_width=int(round(W * highres_scale / 64.0) * 64),
                target_height=int(round(H * highres_scale / 64.0) * 64)
            ) for p in pixels
        ]

        pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor

        memory_management.load_models_to_gpu([unet])
        latents = latents.to(device=unet.device, dtype=unet.dtype)

        latents = pipeline(
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

        memory_management.load_models_to_gpu([vae])
        latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
        pixels = vae.decode(latents).sample
        pixels = pytorch2numpy(pixels)

    return [Image.fromarray(p) for p in pixels]

def omost_generate(image_save_path, message, num_samples, seed, image_width, image_height, highres_scale, steps, cfg, highres_steps, highres_denoise, negative_prompt):
    # You can batch process, or output a single pass, please input message as str or list[str].
    if not isinstance(message, list):
        message = [message]

    for m in message:
        # Using llm to generate canvus code for adding global and local descriptions
        response = llm_output(m, seed)
        # Processing the response to generate canvus outputs
        canvas_outputs = post_process(response, m, seed)
        # Image render
        images = diffusion_output(canvas_outputs, num_samples, seed, image_width, image_height, highres_scale, steps, cfg, highres_steps, highres_denoise, negative_prompt)
        # Save images to image_save_path
        for i in range(len(images)):
            file_name = os.path.join(image_save_path, f"{sanitize_filename(m)}_{i}.png")
            images[i].save(file_name)
        print(f"The topic is: {m}; the image save path is:{image_save_path}")

if __name__ == "__main__":
    # default settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_save_path', type=str, default='/path/to/save_path', help= "save path for generated images")
    parser.add_argument('--message', type=str, default="the beautiful city", help= "the input for llm model")
    parser.add_argument('--num_samples', type=int, default=2, help= "one batch image nums")
    parser.add_argument('--seed', type=int, default=12345, help= "seed")
    parser.add_argument('--image_width', type=int, default=896, help= "the image width")
    parser.add_argument('--image_height', type=int, default=1152, help= "the image width")
    parser.add_argument('--highres_scale', type=float, default=2, help= "the scale of image highres")
    parser.add_argument('--steps', type=int, default=30, help= "the num steps for diffusion process")
    parser.add_argument('--cfg', type=float, default=7.0, help= "CFG Scale")
    parser.add_argument('--highres_steps', type=int, default=20, help= "the steps of image highres")
    parser.add_argument('--highres_denoise', type=float, default=0.4, help= "the denoise strength of image highres")
    parser.add_argument('--negative_prompt', type=str, default="lowres, bad anatomy, bad hands, cropped, worst quality", help= "negative prompt")
    # super resolution settings https://github.com/xinntao/Real-ESRGAN
    parser.add_argument('--model_name',type=str,default='RealESRGAN_x4plus',help=('Model names: RealESRGAN_x4plus | RealESRGAN_x4plus_anime_6B'))
    parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--fp32', type=bool, default=True, help='Use fp32 precision during inference. To False: fp16 (half precision).')

    args = parser.parse_args()

    # SDXL
    # sdxl_name = 'SG161222/RealVisXL_V4.0'
    # sdxl_name = 'stabilityai/stable-diffusion-xl-base-1.0'
    sdxl_name = '/path/to/RealVisXL_V4.0'
    tokenizer = CLIPTokenizer.from_pretrained(
        sdxl_name, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        sdxl_name, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(
        sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
    text_encoder_2 = CLIPTextModel.from_pretrained(
        sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
    vae = AutoencoderKL.from_pretrained(
        sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16") # bfloat16 vae
    unet = UNet2DConditionModel.from_pretrained(
        sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16")

    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())

    pipeline = StableDiffusionXLOmostPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
    )
    memory_management.unload_all_models([text_encoder, text_encoder_2, vae, unet])

    # LLM
    # llm_name = 'lllyasviel/omost-phi-3-mini-128k-8bits'
    # llm_name = 'lllyasviel/omost-llama-3-8b-4bits'
    # llm_name = 'lllyasviel/omost-dolphin-2.9-llama3-8b-4bits'
    llm_name = '/path/to/omost-llama-3-8b-4bits'
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.bfloat16,  # This is computation type, not load/memory type. The loading quant type is baked in config.
        device_map="auto"  # This will load model to gpu with an offload system
    )

    llm_tokenizer = AutoTokenizer.from_pretrained(
        llm_name
    )
    memory_management.unload_all_models(llm_model)

    # add RealESRGAN super resolution module
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        sr_netscale = 4
        # https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
        sr_model_path = "/path/to/RealESRGAN_x4plus.pth"
    else:  # RealESRGAN_x4plus_anime_6B  # x4 RRDBNet model with 6 blocks
        sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        sr_netscale = 4
        # https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
        sr_model_path = "/path/to/RealESRGAN_x4plus_anime_6B.pth" 
    
    # super resolution module
    upsampler = RealESRGANer(
        scale=sr_netscale,
        model_path=sr_model_path,
        dni_weight=None,
        model=sr_model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=None)

    # If you want to batch process prompts, please organize the prompts into a list form list[str]
    # you also can input single prompt as str, to generate your wanted image
    # We integrate the entire process into omost_generate method: txt input -> llm -> canvus outputs -> txt2img -> sr -> img2img -> img output
    omost_generate(args.image_save_path, 
                args.message, 
                args.num_samples, 
                args.seed, 
                args.image_width, 
                args.image_height, 
                args.highres_scale, 
                args.steps, 
                args.cfg, 
                args.highres_steps, 
                args.highres_denoise, 
                args.negative_prompt
            )
