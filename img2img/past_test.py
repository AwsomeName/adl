import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

sdpath = "/home/lc/cv_models/stable-diffusion-v1-5"
sdpath = "/home/lc/cv_models/stable-diffusion-2-1"
pipeline = AutoPipelineForImage2Image.from_pretrained(
    sdpath, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    # "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

# prepare image
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image('/home/lc/code/ADL/png/雷神3.png')
# init_image = load_image('/home/lc/code/ADL/png/img2img-init.png')

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# prompt = "the girl is cooking, 8k"
prompt = "The girl in the kitchen, detailed, gentle, 8k"
# prompt = "Astronauts in the jungle, cool tones, soft colors, detailed, 8k"
negative_prompt = "Ugly, deformed, disfigured, with poor details and anatomical structure"

# pass prompt and image to pipeline
image = pipeline(
    prompt, 
    image=init_image,
    guiding_scale= 10.0,
    strength= 1.0,
    negative_prompt=negative_prompt,
    ).images[0]
make_image_grid([init_image, image], rows=1, cols=2).save("img2img.png")