from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image, make_image_grid

# 抽取pose
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
original_image = load_image(
    "/home/lc/code/ADL/png/雷神_副本.png",
    # "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"
)
openpose_image = openpose(original_image)
pose_png = make_image_grid([original_image, openpose_image], rows=1, cols=2)
pose_png.save("gen_pose_test.png")


# 加载模型
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
import torch

controlnets = [
    ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
    ),
    ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True
    ),
]

# 下面两个模型在一个文件夹
sd_path = "/home/lc/cv_models/stable-diffusion-xl-base-1.0"
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnets, vae=vae, torch_dtype=torch.float16, use_safetensors=True
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()


# 生成图像
prompt = "a giant standing in a fantasy landscape, best quality"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

generator = torch.manual_seed(1)

images = [openpose_image.resize((1024, 1024)), canny_image.resize((1024, 1024))]
# images = [openpose_image.resize((1024, 1024)), pose_png.resize((1024, 1024))]

images = pipe(
    prompt,
    image=images,
    num_inference_steps=25,
    generator=generator,
    negative_prompt=negative_prompt,
    num_images_per_prompt=3,
    controlnet_conditioning_scale=[1.0, 0.8],
).images
make_image_grid([original_image, canny_image, openpose_image,
# make_image_grid([original_image, pose_png, openpose_image,
                images[0].resize((512, 512)), images[1].resize((512, 512)), images[2].resize((512, 512))], rows=2, cols=3)



