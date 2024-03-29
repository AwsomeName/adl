from huggingface_hub import hf_hub_download
from PIL import Image
import imageio

# https://github.com/Picsart-AI-Research/Text2Video-Zero/blob/main/__assets__/poses_skeleton_gifs/dance1_corr.mp4
# filename = "__assets__/poses_skeleton_gifs/dance1_corr.mp4"
# repo_id = "PAIR/Text2Video-Zero"
# video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)
video_path = "/home/lc/code/ADL/videos/dance1_corr.mp4"

reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
pose_images = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]


import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

model_id = "runwayml/stable-diffusion-v1-5"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

pipeline.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipeline.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))


latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(len(pose_images), 1, 1, 1)

prompt = "Darth Vader dancing in a desert"
result = pipeline(prompt=[prompt] * len(pose_images), image=pose_images, latents=latents).images
imageio.mimsave("video.mp4", result, fps=4)