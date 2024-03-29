from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np

original_image = load_image(
    "/home/lc/code/ADL/test_control_net/input_image_vermeer.png"
    # "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)

image = np.array(original_image)

low_threshold = 100
high_threshold = 200

# 抽取canny图像
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
canny_image.save("canny_image.png")

# 根据canny图像生成
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

# canny_path = "/home/lc/sv_models/sd-controlnet-canny"
canny_path = "/home/lc/cv_models/sd-controlnet-canny"
# sd_path = "/home/lc/sv_models/stable-diffusion-2-1"
sd_path = "/home/lc/cv_models/stable-diffusion-2-1"
# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
controlnet = ControlNetModel.from_pretrained(canny_path, torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    sd_path, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    # "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

# 改为蒙娜丽莎
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

output = pipe(
    "the mona lisa", image=canny_image
).images[0]
make_image_grid([original_image, canny_image, output], rows=1, cols=3).save("new_gen.png")

