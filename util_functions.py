import matplotlib.pyplot as plt
import numpy as np

COPPER = True


def plot_image(arr: np.ndarray, title="No Title") -> None:
    """
    Plots an image from the dataset
    """
    plt.axis("off")
    if COPPER:
        plt.imshow(arr, cmap="copper")
    else:
        plt.imshow(arr)
    plt.title(f"{title} (size: {arr.shape})")
    plt.show()


def plot_x_y(x: np.ndarray, y: np.ndarray, title="No Title") -> None:
    """
    Plots two images next to each other
    """
    plt.figure(figsize=(6, 2))

    plt.subplot(121)
    plt.axis("off")
    if COPPER:
        plt.imshow(x, interpolation="none", cmap="copper")
    else:
        plt.imshow(x, interpolation="none")
    plt.title(f"x {x.shape}")

    plt.subplot(122)
    plt.axis("off")
    if COPPER:
        plt.imshow(x, interpolation="none", cmap="copper")
    else:
        plt.imshow(x, interpolation="none")
    plt.title(f"y {y.shape}")

    plt.tight_layout()
    plt.show()
