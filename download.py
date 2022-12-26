import argparse
from pathlib import Path

import gdown


class Constant:
    SyntheticURL = "https://drive.google.com/drive/folders/1YRP6nt96OJUOzEYY6N_Qh5xb3wEVFSjg"


class Experiment:
    Synthetic = "synthetic"
    ImageNet = "imagenet"
    All = [Synthetic, ImageNet]


def download_synthetic_checkpoints():
    """
    Creates a new folder under `models/synthetic`. Then, downloads the synthetic
    model checkpoints into the folder.
    """
    path = "models/synthetic"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(Constant.SyntheticURL, output=path,
                          quiet=False, use_cookies=False)


def download_imagenet_checkpoints():
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="synthetic",
        help="Which experiment in the DDIBs paper to run."
    )
    args = parser.parse_args()

    exp = args.experiment
    assert args.experiment.lower() in Experiment.All

    print(f"Configuring the project for {exp} experiments.")
    if exp == Experiment.Synthetic:
        download_synthetic_checkpoints()


if __name__ == "__main__":
    main()
