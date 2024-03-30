import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
# from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda"
dtype = torch.float16

# step = 4  # Options: [1,2,4,8]
step = 4  # Options: [1,2,4,8]
repo = "/home/lc/cv_models/AnimateDiff-Lightning"
# repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
# base = "emilianJR/epiCRealism"  # Choose to your favorite base model.
base = "/home/lc/cv_models/epiCRealism"

file_path = "/home/lc/cv_models/AnimateDiff-Lightning/animatediff_lightning_4step_diffusers.safetensors"
adapter = MotionAdapter().to(device, dtype)
# adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))
# adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt, local_dir="/home/lc/data_b/CV/", local_files_only=True), device=device))
adapter.load_state_dict(load_file(file_path, device=device))
print("load adapter done")
pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
print("load model done")
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, 
    timestep_spacing="trailing", 
    beta_schedule="linear")
print("load scheduler done")

print("start pip")
output = pipe(
    prompt="A girl smiling", 
    # prompt="原神角色雷神在做饭", 
    # prompt="The role of Genshin Impact: Lei Shen is cooking",
    guidance_scale=1.0, 
    num_inference_steps=step)
print("start gen gif ")
export_to_gif(output.frames[0], "animation.gif")

