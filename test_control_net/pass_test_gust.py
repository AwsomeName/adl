from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image, make_image_grid
import numpy as np
import torch
from PIL import Image
import cv2

cnpath = "/home/lc/cv_models/sd-controlnet-canny"
sdpath = "/home/lc/cv_models/stable-diffusion-v1-5"
controlnet = ControlNetModel.from_pretrained(cnpath, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(sdpath, controlnet=controlnet, use_safetensors=True).to("cuda")
# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", use_safetensors=True)
# pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, use_safetensors=True).to("cuda")

# original_image = load_image("https://huggingface.co/takuma104/controlnet_dev/resolve/main/bird_512x512.png")
original_image = load_image("/home/lc/code/ADL/png/测试.png")

image = np.array(original_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

image = pipe("", image=canny_image, guess_mode=True, guidance_scale=3.0).images[0]
make_image_grid([original_image, canny_image, image], rows=1, cols=3).save("gust.png")

# 猜测模式