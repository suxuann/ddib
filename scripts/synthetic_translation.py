"""
Synthetic domain translation.
"""

import argparse
import numpy as np
import os
import pathlib
import torch.distributed as dist

from common import (
    read_model_and_diffusion, get_code_and_dataset_folders
)
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults_2d,
    add_dict_to_argparser,
)
from guided_diffusion.synthetic_datasets import (
    scatter,
    heatmap,
    load_2d_data,
    Synthetic2DType,
)


def main():
    args = create_argparser().parse_args()
    logger.log(f"args: {args}")

    dist_util.setup_dist()
    logger.configure()
    logger.log("starting to sample synthetic data.")

    code_folder, data_folder = get_code_and_dataset_folders()
    image_folder = os.path.join(code_folder, f"log2DImages")
    pathlib.Path(image_folder).mkdir(parents=True, exist_ok=True)

    shapes = list(Synthetic2DType)

    models = []
    diffusion = None
    logger.log(f"reading models for synthetic data...")
    for i in range(len(shapes)):
        log_dir = os.path.join(code_folder, f"log2D{i}")
        model, diffusion = read_model_and_diffusion(args, log_dir)
        models.append(model)

    # Translation pairs. The values 0-5 are indexes for the list [Moons, Checkerboards, CR, CS, PR, PS].
    translation_pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (1, 3), (1, 4),
                         (2, 4), (4, 2), (5, 3), (2, 3), (4, 5)]
    for (i, j) in translation_pairs:
        source_model = models[i]
        shape_s = shapes[i]
        target_model = models[j]
        shape_t = shapes[j]

        image_subfolder = os.path.join(image_folder, f"translation_{i}_{j}")
        pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

        sources = []
        latents = []
        targets = []
        data = load_2d_data(n_samples=90000, batch_size=args.batch_size, shape=shape_s, training=False)

        for k, (source, extra) in enumerate(data):
            logger.log(f"translating {i}->{j}, batch {k}, shape {source.shape}...")
            source = source.to(dist_util.dev())

            noise = diffusion.ddim_reverse_sample_loop(
                source_model, source,
                clip_denoised=False,
                device=dist_util.dev(),
            )
            logger.log(f"obtained latent representation for {source.shape[0]} samples...")
            logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")

            target = diffusion.ddim_sample_loop(
                target_model, (args.batch_size, 2),
                noise=noise,
                clip_denoised=False,
                device=dist_util.dev(),
            )
            logger.log(f"finished translation {target.shape}")
            sources.append(source.cpu().numpy())
            latents.append(noise.cpu().numpy())
            targets.append(target.cpu().numpy())

        sources = np.concatenate(sources, axis=0)
        sources_path = os.path.join(image_subfolder, 'source.npy')
        np.save(sources_path, sources)
        sources_image_path = os.path.join(image_subfolder, 'scatter_source.png')
        scatter(sources, sources_image_path)
        sources_image_path = os.path.join(image_subfolder, 'heatmap_source.png')
        heatmap(sources, sources_image_path)

        latents = np.concatenate(latents, axis=0)
        latents_path = os.path.join(image_subfolder, 'latent.npy')
        np.save(latents_path, latents)
        latents_image_path = os.path.join(image_subfolder, 'scatter_latent.png')
        scatter(latents, latents_image_path)
        latents_image_path = os.path.join(image_subfolder, 'heatmap_latent.png')
        heatmap(latents, latents_image_path)

        targets = np.concatenate(targets, axis=0)
        targets_path = os.path.join(image_subfolder, 'target.npy')
        np.save(targets_path, targets)
        targets_image_path = os.path.join(image_subfolder, 'scatter_target.png')
        scatter(targets, targets_image_path)
        targets_image_path = os.path.join(image_subfolder, 'heatmap_target.png')
        heatmap(targets, targets_image_path)

        dist.barrier()
        logger.log(f"synthetic data translation complete: {i}->{j}\n\n")


def create_argparser():
    defaults = dict(
        num_samples=80000,
        batch_size=20000,
        model_path=""
    )
    defaults.update(model_and_diffusion_defaults_2d())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
