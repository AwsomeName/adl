import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from safetensors.torch import load_file

ad_path = "/home/lc/cv_models/AnimateDiff-Lightning/animatediff_lightning_4step_diffusers.safetensors"
# ad_path = "/home/lc/cv_models/AnimateDiff-Lightning/"
# adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

device = "cuda"
dtype = torch.float16
adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(ad_path, device=device))
# adapter = MotionAdapter.from_pretrained(ad_path, torch_dtype=torch.float16)

base = "/home/lc/cv_models/epiCRealism"
# pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
pipeline = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
    # "emilianJR/epiCRealism",
    base,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipeline.scheduler = scheduler
pipeline.enable_vae_slicing()
pipeline.enable_model_cpu_offload()


output = pipeline(
    prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
    negative_prompt="bad quality, worse quality, low resolution",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=5,
    # num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(49),
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")