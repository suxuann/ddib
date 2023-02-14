"""Sample from the synthetic models and save the results as png."""

import argparse
import os
import pathlib

import numpy as np
import torch
import torch.distributed as dist

from common import read_model_and_diffusion
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import model_and_diffusion_defaults_2d, add_dict_to_argparser
from guided_diffusion.synthetic_datasets import Synthetic2DType, scatter, heatmap


def main():
    args = create_argparser().parse_args()
    logger.log(f"args: {args}")

    dist_util.setup_dist()
    logger.configure()
    logger.log("starting to sample synthetic data.")

    code_folder = os.getcwd()
    image_folder = os.path.join(code_folder, f"experiments/images")
    pathlib.Path(image_folder).mkdir(parents=True, exist_ok=True)

    shapes = list(Synthetic2DType)
    i = args.task
    shape = shapes[i]

    logger.log("creating 2d model and diffusion...")
    model_dir = os.path.join(code_folder, f"models/synthetic/log2D{i}")
    model, diffusion = read_model_and_diffusion(args, model_dir)

    all_samples = []
    while len(all_samples) * args.batch_size < args.num_samples:
        logger.log("sampling in progress.")
        logger.log(f"device: {dist_util.dev()}")

        sample = diffusion.ddim_sample_loop(
            model, (args.batch_size, 2),
            clip_denoised=False,
            device=dist_util.dev(),
        )
        gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

    points = np.concatenate(all_samples, axis=0)
    points = points[:args.num_samples]
    points_path = os.path.join(image_folder, f"points_{shape}.npy")
    np.save(points_path, points)
    scatter_path = os.path.join(image_folder, f"scatter_{shape}.png")
    heatmap_path = os.path.join(image_folder, f"heatmap_{shape}.png")
    scatter(points, scatter_path)
    heatmap(points, heatmap_path)

    dist.barrier()
    logger.log(f"sampling synthetic data complete: {shape}\n\n")


def create_argparser():
    defaults = dict(
        num_samples=80000,
        batch_size=20000,
        model_path=""
    )
    defaults.update(model_and_diffusion_defaults_2d())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=int,
        default=0,
        help="Which dataset to sample from."
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
