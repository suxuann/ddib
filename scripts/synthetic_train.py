"""
Train a diffusion model on 2D synthetic datasets.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import model_and_diffusion_defaults_2d, create_model_and_diffusion_2d, args_to_dict, \
    add_dict_to_argparser
from guided_diffusion.synthetic_datasets import Synthetic2DType, load_2d_data
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    logger.log(f"args: {args}")

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating 2d model and diffusion...")
    model, diffusion = create_model_and_diffusion_2d(
        **args_to_dict(args, model_and_diffusion_defaults_2d().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating 2d data loader...")
    data = load_2d_data(
        batch_size=args.batch_size,
        shape=list(Synthetic2DType)[args.task]
    )

    logger.log("training 2d model...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        task=0,  # 0 to 5 inclusive
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=30000,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults_2d())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
