import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image
# image = cv2.imread('a.jpg')
# image = cv2.imread('b.jpg')
image = cv2.imread('c.jpg')

def rgb_to_grayscale(image):
    """Step 2a: Convert BGR (OpenCV default) to Grayscale using weighted sum"""
    B, G, R = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    grayscale = 0.299 * R + 0.587 * G + 0.114 * B
    return grayscale.astype(np.uint8)

def gaussian_smoothing(image_gray, sigma=1.0):
    """Step 2b: Apply Gaussian smoothing"""
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1  # kernel size must be odd
    smoothed = cv2.GaussianBlur(image_gray, (ksize, ksize), sigma)
    return smoothed

def sobel_edge_detection(image_smoothed, threshold=100):
    """Step 3: Sobel edge detection with thresholding"""
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])

    Ix = cv2.filter2D(image_smoothed, cv2.CV_64F, Gx)
    Iy = cv2.filter2D(image_smoothed, cv2.CV_64F, Gy)

    magnitude = np.sqrt(Ix**2 + Iy**2)
    direction = np.arctan2(Iy, Ix)

    edge_map = (magnitude > threshold).astype(np.uint8) * 255
    return edge_map, magnitude, direction

# Example usage
if __name__ == "__main__":
    gray = rgb_to_grayscale(image)
    smoothed = gaussian_smoothing(gray, sigma=1.5)
    edges, mag, theta = sobel_edge_detection(smoothed, threshold=100)

    # Display results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(gray, cmap='gray'); plt.title("Grayscale"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(smoothed, cmap='gray'); plt.title("Smoothed"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(edges, cmap='gray'); plt.title("Edges"); plt.axis('off')
    plt.tight_layout()
    plt.show()
