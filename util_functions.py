import matplotlib.pyplot as plt
import numpy as np


def plot_image(arr: np.ndarray, title="No Title") -> None:
    """
    Plots an image from the dataset
    """
    plt.axis("off")
    plt.imshow(arr, cmap="copper")
    plt.title(f"{title} (size: {arr.shape})")
    plt.show()


def plot_x_y(x: np.ndarray, y: np.ndarray, title="No Title") -> None:
    """
    Plots two images next to each other
    """
    plt.figure(figsize=(6, 2))

    plt.subplot(121)
    plt.axis("off")
    plt.imshow(x, interpolation="none", cmap="copper")
    plt.title(f"x {x.shape}")

    plt.subplot(122)
    plt.axis("off")
    plt.imshow(y, interpolation="none", cmap="copper")
    plt.title(f"y {y.shape}")

    plt.tight_layout()
    plt.show()
