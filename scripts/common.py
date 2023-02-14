import os

import numpy as np
import torch as th
import torch.distributed as dist
from skimage import io

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
)
from guided_diffusion.script_util import (
    model_and_diffusion_defaults_2d,
    create_model_and_diffusion_2d,
    create_model_and_diffusion,
    args_to_dict,
)


def get_latest_model_path_in_directory(directory):
    """Returns the path to the latest model in the given directory."""
    model_files = [file for file in os.listdir(directory) if file.startswith("model")]
    model_numbers = sorted([int(file[5:-3]) for file in model_files])
    if len(model_numbers) == 0:
        return ""
    model_number = str(f"{model_numbers[-1]}").zfill(6)
    model_file = f"model{model_number}.pt"
    model_path = os.path.join(directory, model_file)
    return model_path, model_number


def read_model_and_diffusion(args, log_dir, synthetic=True):
    """Reads the latest model from the given directory."""
    model_path, _ = get_latest_model_path_in_directory(log_dir)
    logger.log(f"Model path: {model_path}")

    if synthetic:
        model, diffusion = create_model_and_diffusion_2d(
            **args_to_dict(args, model_and_diffusion_defaults_2d().keys()),
        )
    else:
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys()),
        )
    model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    return model, diffusion


def sample_to_images(sample):
    """
    :param sample: (N, 3, H, W) output from trained DDIM model
    :return images: list of (H, W, 3)
    """
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_samples, sample)
    images = [sample.cpu().numpy() for sample in gathered_samples]
    return images


def normalize_image(image):
    return image.astype(np.float32) / 127.5 - 1


def compute_mse(XA, XB):
    return np.square(np.subtract(XA, XB)).mean()


def compute_mse_on_images(path_A, path_B):
    XA = normalize_image(io.imread(path_A))
    XB = normalize_image(io.imread(path_B))
    return compute_mse(XA, XB)
