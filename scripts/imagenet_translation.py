"""
Class-conditional image translation from one ImageNet class to another.
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import (
    load_source_data_for_domain_translation,
    get_image_filenames_for_label
)
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def copy_imagenet_dataset(val_dir, classes):
    """
    Finds the validation images for the given classes from val_dir,
    and copies them over to ./experiments/imagenet for translation.
    """
    base_dir = os.path.join(os.getcwd(), "experiments", "imagenet")
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    for source_label in classes:
        logger.log(f"Copying image files for class {source_label}.")
        filenames = get_image_filenames_for_label(source_label)
        for i, filename in enumerate(filenames):
            path = os.path.join(val_dir, filename)
            copy_path = os.path.join(base_dir, f"{source_label}_{i + 1}.PNG")
            shutil.copyfile(path, copy_path)


def main():
    args = create_argparser().parse_args()
    logger.log(f"arguments: {args}")

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    # Copies the source dataset from ImageNet validation set.
    logger.log("copying source dataset.")
    source = [int(v) for v in args.source.split(",")]
    target = [int(v) for v in args.target.split(",")]
    source_to_target_mapping = {s: t for s, t in zip(source, target)}
    copy_imagenet_dataset(args.val_dir, source)

    logger.log("running image translation...")
    data = load_source_data_for_domain_translation(
        batch_size=args.batch_size,
        image_size=args.image_size,
        classes=source
    )

    for i, (batch, extra) in enumerate(data):
        logger.log(f"translating batch {i}, shape {batch.shape}.")

        logger.log("saving the original, cropped images.")
        images = ((batch + 1) * 127.5).clamp(0, 255).to(th.uint8)
        images = images.permute(0, 2, 3, 1)
        images = images.contiguous()
        images = images.cpu().numpy()
        for index in range(images.shape[0]):
            filepath = extra["filepath"][index]
            image = Image.fromarray(images[index])
            image.save(filepath)
            logger.log(f"    saving: {filepath}")

        batch = batch.to(dist_util.dev())

        # Class labels for source and target sets
        source_y = dict(y=extra["y"].to(dist_util.dev()))
        target_y_list = [source_to_target_mapping[v.item()] for v in extra["y"]]
        target_y = dict(y=th.tensor(target_y_list).to(dist_util.dev()))

        # First, use DDIM to encode to latents.
        logger.log("encoding the source images.")
        noise = diffusion.ddim_reverse_sample_loop(
            model_fn,
            batch,
            clip_denoised=False,
            model_kwargs=source_y,
            device=dist_util.dev(),
        )
        logger.log(f"obtained latent representation for {batch.shape[0]} samples...")
        logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")

        # Next, decode the latents to the target class.
        sample = diffusion.ddim_sample_loop(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=target_y,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            eta=args.eta
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        images = []
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(images) * args.batch_size} samples")

        logger.log("saving translated images.")
        images = np.concatenate(images, axis=0)

        for index in range(images.shape[0]):
            base_dir, filename = os.path.split(extra["filepath"][index])
            filename, ext = filename.split(".")
            filepath = os.path.join(base_dir, f"{filename}_translated_{target_y_list[index]}.{ext}")
            image = Image.fromarray(images[index])
            image.save(filepath)
            logger.log(f"    saving: {filepath}")

    dist.barrier()
    logger.log(f"domain translation complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=8,
        classifier_scale=1.0,
        eta=0.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/imagenet/256x256_diffusion.pt",
        help="Path to the diffusion model weights."
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        default="./models/imagenet/256x256_classifier.pt",
        help="Path to the classifier model weights."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="260,261,282,283",
        help="Source domains."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="261,262,283,284",
        help="Target domains."
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="",
        help="The local directory containing ImageNet validation dataset, "
             "containing filenames like ILSVRC2012_val_000XXXXX.JPG."
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
