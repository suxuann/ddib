import enum

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, Dataset


def heatmap(points, filename='heatmap.png'):
    """
    Draws a heatmap of the 2D distribution underlying points.
    https://stackoverflow.com/questions/2369492/generate-a-heatmap-in-matplotlib-using-a-scatter-data-set
    :param points: (N, 2)
    :param filename
    """
    x, y = np.rollaxis(points, 1)
    axis_lim = 3
    axis_bins = np.linspace(-axis_lim, axis_lim, 100)

    plt.gca().set_aspect('equal')
    plt.hist2d(x, y, bins=[axis_bins, axis_bins])
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', transparent=True, pad_inches=0)


def scatter(points, filename='scatter.png', enable_color_interpolation=True):
    """Draws a scatter, fine plots of the given points.
    TODO Set the x,y limits."""
    xlim, ylim = 3, 3
    N = points.shape[0]
    px, py = np.rollaxis(points, 1)
    plt.figure(figsize=(6, 6))
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)
    ax = plt.gca()
    ax.set_aspect('equal')
    # ax.set_facecolor('#430154')
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.xticks([]), plt.yticks([])
    if enable_color_interpolation:
        # colors = np.linspace(-np.pi, np.pi, N)  # Color interpolation
        colors = np.arange(0, N)
    else:
        colors = np.array([[0.525776, 0.833491, 0.288127]])
    plt.scatter(px, py, s=1.0, marker='o', c=colors, cmap='rainbow', linewidths=0, alpha=1.0)
    plt.savefig(filename, bbox_inches='tight', transparent=False, pad_inches=0)


def make_checkerboard(n_samples):
    x1 = np.random.rand(n_samples) * 4 - 2
    x1 = np.sort(x1)[::-1]
    x2 = np.random.rand(n_samples) - np.random.randint(0, 2, n_samples) * 2
    x2 = x2 + (np.floor(x1) % 2)
    return np.stack([x1, x2], axis=1)


def normalize(points):
    """
    Normalize so that the points have zero mean and unit variance.
    :param points: (N, 2)
    """
    mean = points.mean()
    std = points.std()
    return (points - mean) / std


class BaseSampler(object):
    def __init__(self, radii: np.array, centers: np.array, width: float):
        """
        Base sampler for rings and squares.
        :param radii: radius of the rings or squares
        :param centers
        :param width: the width of each ring / square
        """
        self.num_objects = radii.shape[0]  # Number of rings or squares
        self.radii = np.array(radii, np.double)
        self.centers = np.array(centers, np.double)
        self.width = width


class RingSampler(BaseSampler):
    """
    Data sampler class to generate synthetic 2D distribution data.
    The implementation considers only circle-like distributions.
    """

    def sample(self, N):
        """
        Samples from the 2D distribution.
        Returns a sample of the shape (N, 2).
        :param N: number of data points
        :return:
        """
        # Assigns points to one of the K rings
        K = self.num_objects
        assert N % K == 0
        indices = np.arange(0, K)
        indices = np.tile(indices, N // K + 1)[:N]
        indices = np.sort(indices)  # (0,0,0,0,1,1,1,1...
        assert indices.shape[0] == N
        centers = self.centers[indices]

        radii_eps = (np.random.rand(N) - .5) * self.width
        radii = self.radii[indices] + radii_eps

        # Randomly assigns points on the ring
        theta = np.random.rand(K, N // K) * 2 * np.pi
        theta = np.sort(theta, 1).reshape(-1)[::-1]
        x, y = np.rollaxis(centers, 1)
        px = radii * np.cos(theta) + x
        py = radii * np.sin(theta) + y

        points = np.stack([px, py], axis=1)
        points = normalize(points)
        return points, indices


class ConcentricRingSampler(RingSampler):
    def __init__(self, radii: np.array, width: float = 0.5):
        num_objects = radii.shape[0]
        centers = np.zeros((num_objects, 2))
        super(ConcentricRingSampler, self).__init__(radii, centers, width)


class OlympicRingSampler(RingSampler):
    def __init__(self, radii: np.array = np.ones(5), width: float = 0.5):
        # The ratio between radii and centers is fixed
        num_objects = radii.shape[0]
        centers = np.array([(-140, 0), (0, 0), (140, 0), (-55, -50), (55, -50)], np.float32) / float(50)
        centers = centers[:num_objects]
        super(OlympicRingSampler, self).__init__(radii, centers, width)


class SquareSampler(BaseSampler):

    def sample(self, N):
        # Generate numbers for the four edges
        K = self.num_objects
        assert N % (4 * K) == 0
        rand = lambda c: 2 * np.sort(np.random.rand(c) - 0.5)  # Increasing in [-1, 1]

        px = list()
        py = list()
        indices = list()

        for i in range(K):
            C = N // (K * 4)
            px0 = rand(C)  # Up
            py0 = np.ones(C)
            px1 = np.ones(C)  # Right
            py1 = rand(C)[::-1]
            px2 = rand(C)[::-1]  # Down
            py2 = -np.ones(C)
            px3 = -np.ones(C)  # Left
            py3 = rand(C)
            indices.append(np.ones(C * 4) * i)
            px += [px0, px1, px2, px3]
            py += [py0, py1, py2, py3]
        px = np.concatenate(px)
        py = np.concatenate(py)
        indices = np.concatenate(indices).astype(int)

        # Then, assign the points randomly to the squares
        radii_eps = (np.random.rand(N) - .5) * self.width
        radii = self.radii[indices] + radii_eps

        px, py = px * radii, py * radii
        points = np.stack([px, py], axis=1)

        centers = self.centers[indices]
        points = points + centers
        points = normalize(points)
        return points, indices


class ConcentricSquareSampler(SquareSampler):
    def __init__(self, radii: np.array, width: float = 0.5):
        num_objects = radii.shape[0]
        centers = np.zeros((num_objects, 2))
        super(ConcentricSquareSampler, self).__init__(radii, centers, width)


class OlympicSquareSampler(SquareSampler):
    def __init__(self, radii: np.array = np.ones(5), width: float = 0.5):
        # The ratio between radii and centers is fixed
        num_objects = radii.shape[0]
        centers = np.array([(-150, 0), (0, 0), (150, 0), (-55, -50), (55, -50)], np.float32) / float(50)
        centers = centers[:num_objects]
        super(OlympicSquareSampler, self).__init__(radii, centers, width)


class Synthetic2DType(enum.Enum):
    """Which type of synthetic 2D datasets."""
    MOONS = enum.auto()
    CHECKERBOARD = enum.auto()
    CONCENTRIC_RINGS = enum.auto()
    CONCENTRIC_SQUARES = enum.auto()
    OLYMPIC_RINGS = enum.auto()  # The name in the paper is Parallel Rings
    OLYMPIC_SQUARES = enum.auto()


class Synthetic2DDataset(Dataset):
    def __init__(self, n_samples, shape: Synthetic2DType):
        super().__init__()
        self.n_samples = n_samples
        self.shape = shape
        self.shape_val = shape.value

        noise = 0.05
        radii_increasing = np.array([1, 2, 3])
        radii_uniform = np.array([1, 1, 1])
        if shape == Synthetic2DType.CONCENTRIC_RINGS:
            points, indices = ConcentricRingSampler(radii_increasing).sample(n_samples)
        elif shape == Synthetic2DType.OLYMPIC_RINGS:
            points, indices = OlympicRingSampler(radii_uniform).sample(n_samples)
        elif shape == Synthetic2DType.CONCENTRIC_SQUARES:
            points, indices = ConcentricSquareSampler(radii_increasing).sample(n_samples)
        elif shape == Synthetic2DType.OLYMPIC_SQUARES:
            points, indices = OlympicSquareSampler(radii_uniform).sample(n_samples)
        elif shape == Synthetic2DType.MOONS:
            points, labels = make_moons(n_samples=n_samples, shuffle=False, noise=noise)
        elif shape == Synthetic2DType.CHECKERBOARD:
            points = make_checkerboard(n_samples=n_samples)
        else:
            raise NotImplementedError
        self.points = normalize(points).astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.points[index], {}


def load_2d_data(batch_size, shape: Synthetic2DType, training=True, n_samples=300000):
    dataset = Synthetic2DDataset(n_samples=n_samples, shape=shape)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, drop_last=False)
    if training:
        while True:
            yield from loader
    else:
        yield from loader


def save_plots():
    """Draws plots of the 2D distributions."""
    N = 150000
    for shape in list(Synthetic2DType):
        print(f"Saving {shape} images.")
        dataset = Synthetic2DDataset(N, shape)
        points = dataset.points
        print(points.mean(), points.std())
        heatmap(points, filename=f"synthetic_images/{shape.value}_{shape}_Heatmap.png")
        scatter(points, filename=f"synthetic_images/{shape.value}_{shape}_Scatter.png")


if __name__ == '__main__':
    save_plots()
