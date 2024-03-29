import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

sd_path = "/home/lc/cv_models/stable-video-diffusion-img2vid-xt-1-1"
pipeline = StableVideoDiffusionPipeline.from_pretrained(
    sd_path, torch_dtype=torch.float16, variant="fp16"
    # "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()

# image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = load_image("/home/lc/code/ADL/png/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipeline(image, decode_chunk_size=8, generator=generator).frames[0]
g_mp4 = export_to_video(frames, "generated.mp4", fps=7)
g_mp4.save("./test.mp4")