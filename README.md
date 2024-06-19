# Omost_with_SR

Omost_with_SR is a project forked by [Omost](https://github.com/lllyasviel/Omost), enhanced with [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) to provide super-resolution capabilities for improved image quality. We extend our gratitude to the contributors of both projects. 

+ If `highres_scale > 1`, the model will adopt the super resolution function, the entire process is: 
**text inputs -> llm -> Canvus outputs -> txt2img -> sr -> img2img -> image outputs**

+ If `highres_scale = 1`, the model will adopt the super resolution function, the entire process is equal to the origin Omost：
**text inputs -> llm -> Canvus outputs -> txt2img -> image outputs**

In particular, we provide a simple inference code `Omost_generate.py` and OmostAutoPipeline class in `Omost_api.py` to quickly use and debug.

# Get Started
you can use the below deployment (requires 8GB Nvidia VRAM):

    git clone https://github.com/lllyasviel/Omost.git
    cd Omost_with_SR
    conda create -n omost python=3.10
    conda activate omost
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    python gradio_app.py

(Note that quant LLM requires `bitsandbytes` - some 9XX or 10XX or 20XX GPUs may have trouble in running it. If that happens, just use our official huggingface space.)

+ You need to make sure that `bitsandbytes==0.43.1`  or you may have a problem.

+ if `ModuleNotFoundError: No module named‘torchvision.transforms.functional_tensor’`，please revise as `torchvision.transforms._functional_tensor`.