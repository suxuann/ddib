# Dual Diffusion Implicit Bridges

This is the codebase
for [Dual Diffusion Implicit Bridges for Image-to-Image Translation](https://arxiv.org/abs/2203.08382).

This repository is heavily based on the repositories from
OpenAI: [openai/guided-diffusion](https://github.com/openai/guided-diffusion)
and [openai/improved-diffusion](https://github.com/openai/improved-diffusion).

Installation follows the same procedures as in the above repositories.

## Synthetic Models

### Pretrained Models

We release checkpoints for the 2D synthetic models in the paper.

We use indexes 0-5 to refer to the 6 synthetic types,
in: [Moons, Checkerboards, Concentric Rings, Concentric Squares, Parallel Rings, Parallel Squares].

Here is the download link for the model
checkpoints: [Synthetic Models](https://drive.google.com/drive/folders/1YRP6nt96OJUOzEYY6N_Qh5xb3wEVFSjg?usp=sharing)

### Scripts to Reproduce the Experiments

First, change the `CODE_FOLDER` variable in `scripts/common.py`.

#### Training Synthetic Models

`python scripts/synthetic_train.py --num_res_blocks 3 --diffusion_steps 4000 --noise_schedule linear --lr 1e-4 --batch_size 20000 --task 0`

Task is an integer in {0, 1, 2, 3, 4, 5} corresponding to one of the synthetic types. Training each model probably takes
only a few (3-4) hours.

#### Cycle Consistency

`python scripts/synthetic_cycle.py --num_res_blocks 3 --diffusion_steps 4000 --batch_size 30000`

The pairs of datasets that we run experiments on are specified in `cycle_pairs` in `scripts/synthetic_cycle.py` file.

#### Synthetic Translation

`python scripts/synthetic_translation.py --num_res_blocks 3 --diffusion_steps 4000 --batch_size 30000`

Similarly, the pairs of datasets are specified in the `translation_pairs` variable.

## To-do List

* Release pretrained models on AFHQ and yosemite datasets
* Add color translation experiments
* Add scripts to translate between AFHQ, yosemite images