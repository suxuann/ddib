import blobfile as bf
import numpy as np
from PIL import Image
from skimage import color
from torch.utils.data import DataLoader, Dataset


class ColorDataset(Dataset):
    # Constants used for LAB color space
    LAB_OFFSET = np.array([[50., 0., 0.]]).astype(np.float32)
    LAB_DENOMINATOR = np.array([[50., 128., 128.]]).astype(np.float32)
    LAB_UPPER_BOUND = np.array([[100., 127., 127.]]).astype(np.float32)
    LAB_LOWER_BOUND = np.array([[0., -128., -128.]]).astype(np.float32)

    def __init__(self, image_path, use_lab_space=True, resize=False, ratio=0):
        super().__init__()
        self.points, self.image_shape = self.read_image(
            image_path, use_lab_space=use_lab_space, resize=resize, ratio=ratio)
        self.resize = resize
        self.n_samples = self.points.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.points[index], {}

    @classmethod
    def normalize(cls, arr, use_lab_space=True):
        """Normalize the given array to [-1, 1]."""
        assert len(arr.shape) == 2 and arr.shape[-1] == 3
        arr = arr.astype(np.float32)
        if use_lab_space:
            arr = (arr - cls.LAB_OFFSET) / cls.LAB_DENOMINATOR
            return arr
        return arr / 127.5 - 1

    @classmethod
    def unnormalize(cls, arr, use_lab_space=True):
        """Unnormalize arr back to normal image range [0, 255]."""
        assert len(arr.shape) == 2 and arr.shape[-1] == 3
        arr = arr.astype(np.float32)
        if use_lab_space:
            return arr * cls.LAB_DENOMINATOR + cls.LAB_OFFSET
        return (arr + 1) * 127.5

    @classmethod
    def read_image(cls, image_path, use_lab_space=True, resize=False, ratio=0):
        """Reads the image at the given path. Returns (arr, shape):
        points in the image and their shape."""
        max_pixel_val = 255.
        with bf.BlobFile(image_path, "rb") as f:
            image = Image.open(f)
            image.load()
        image = image.convert("RGB")
        arr = np.array(image).astype(np.float32)
        if resize:
            arr = arr[::ratio, ::ratio]
        shape = arr.shape
        if use_lab_space:
            arr = arr / max_pixel_val
            arr = color.rgb2lab(arr)
        arr = arr.reshape(-1, 3)
        arr = cls.normalize(arr, use_lab_space)
        return arr, shape


def sample_to_image(arr, image_shape):
    arr = ColorDataset.unnormalize(arr)
    arr = np.clip(arr, ColorDataset.LAB_LOWER_BOUND, ColorDataset.LAB_UPPER_BOUND)
    arr = color.lab2rgb(arr)
    arr = arr.reshape(image_shape).astype(np.uint8)
    return arr


def rgb_sample_to_image(arr, image_shape):
    arr = np.clip(arr, -1., 1.)
    arr = ColorDataset.unnormalize(arr, use_lab_space=False)
    arr = arr.reshape(image_shape).astype(np.uint8)
    return arr


def load_color_data_for_translation(batch_size, dataset):
    loader = DataLoader(
        dataset, batch_size=batch_size,
        num_workers=1, drop_last=False
    )
    yield from loader


def load_color_data_for_training(batch_size, image_path, use_lab_space, logger):
    dataset = ColorDataset(
        image_path=image_path,
        use_lab_space=use_lab_space
    )
    logger.log(f"dataset length: {len(dataset)}, {dataset.points.shape}")
    logger.log(f"dataset points: {dataset.points}, max: {dataset.points.max(0)}, min: {dataset.points.min(0)}")
    loader = DataLoader(
        dataset, batch_size=batch_size,
        num_workers=1, drop_last=False
    )
    while True:
        yield from loader
