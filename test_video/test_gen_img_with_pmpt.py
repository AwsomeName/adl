import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, load_image

iv_path = "/home/lc/cv_models/i2vgen-xl"
pipeline = I2VGenXLPipeline.from_pretrained(iv_path, torch_dtype=torch.float16, variant="fp16")
# pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
# pipeline.enable_model_cpu_offload()

# image_url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"
image_path = "/home/lc/code/ADL/png/img_0009.png"
image = load_image(image_path).convert("RGB")

prompt = "Papers were floating in the air on a table in the library"
negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
generator = torch.manual_seed(8888)

frames = pipeline(
    prompt=prompt,
    image=image,
    num_inference_steps=50,
    negative_prompt=negative_prompt,
    guidance_scale=9.0,
    generator=generator
).frames[0]
export_to_gif(frames, "i2v.gif")