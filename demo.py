import cv2
import numpy as np
import matplotlib.pyplot as plt
from contour import (
    find_contours_manual,
    compute_area,
    compute_perimeter,
    compute_centroid,
    visualize_contours,
)
from PIL import Image, ImageDraw


# Step 1: Load the image
# image = cv2.imread('a.jpg')
image = cv2.imread('f.jpeg')
# image = cv2.imread('c.jpg')
# image = cv2.imread('d.jpg')
# image = cv2.imread('g.jpeg')


def rgb_to_grayscale(image):
    B, G, R = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    grayscale = 0.299 * R + 0.587 * G + 0.114 * B
    return grayscale.astype(np.uint8)

# def gaussian_smoothing(image_gray, sigma=1.0):
#     ksize = int(6 * sigma + 1)
#     if ksize % 2 == 0:
#         ksize += 1
#     smoothed = cv2.GaussianBlur(image_gray, (ksize, ksize), sigma)
#     return smoothed

import numpy as np

def gaussian_kernel(size, sigma):
    """Generate a Gaussian kernel manually."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


def gaussian_smoothing(image_gray, sigma=1.0):
    """Apply Gaussian smoothing manually using convolution."""
    
    # Kernel size rule: ≥ 6*sigma
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    kernel = gaussian_kernel(ksize, sigma)

    # Padding
    pad = ksize // 2
    padded = np.pad(image_gray, ((pad, pad), (pad, pad)), mode='reflect')

    smoothed = np.zeros_like(image_gray, dtype=float)

    # Manual convolution
    for i in range(image_gray.shape[0]):
        for j in range(image_gray.shape[1]):
            region = padded[i:i+ksize, j:j+ksize]
            smoothed[i, j] = np.sum(region * kernel)

    return smoothed.astype(np.uint8)


def sobel_edge_detection(image_smoothed, threshold=50):
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])

    Ix = cv2.filter2D(image_smoothed, cv2.CV_64F, Gx)  # cv2.CV_64F it forves output to be float64
    Iy = cv2.filter2D(image_smoothed, cv2.CV_64F, Gy)

    magnitude = np.sqrt(Ix**2 + Iy**2)
    direction = np.arctan2(Iy, Ix)

    edge_map = (magnitude > threshold).astype(np.uint8) * 255
    return edge_map, magnitude, direction

def find_and_draw_contours(edge_map, original_image):

    contours = find_contours_manual(edge_map)
    return np.array(contour_img), len(contours)



if __name__ == "__main__":
    # Step 1: Convert to grayscale
    gray = rgb_to_grayscale(image)

    # Step 2: Smooth
    smoothed = gaussian_smoothing(gray, sigma=0.65)

    # Step 3: Edge detection
    edges, mag, theta = sobel_edge_detection(smoothed, threshold=100)

    # Step 4: Find contours manually
    contours = find_contours_manual(edges)

    # Step 5: Visualize shaded contours with legend
    visualize_contours(edges, contours)

    print(f"\nTotal contours found: {len(contours)}")

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.title("Original"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(gray, cmap='gray'); plt.title("Grayscale"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(edges, cmap='gray'); plt.title("Edges"); plt.axis('off')
    plt.tight_layout()
    plt.show()
