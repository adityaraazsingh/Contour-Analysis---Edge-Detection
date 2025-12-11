import cv2
import numpy as np
import matplotlib.pyplot as plt
from contour import (
    find_contours_manual,
    compute_area,
    compute_perimeter,
    compute_centroid
)
from PIL import Image, ImageDraw


# Step 1: Load the image
# image = cv2.imread('a.jpg')
# image = cv2.imread('b.jpg')
# image = cv2.imread('c.jpg')
image = cv2.imread('d.jpg')

def rgb_to_grayscale(image):
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

def sobel_edge_detection(image_smoothed, threshold=50):
    """Step 3: Sobel edge detection with thresholding"""
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
    """
    Step 4: Find contours from the edge map and draw them on the original image
    WITHOUT using cv2.
    """

    # 1. Find contours manually
    contours = find_contours_manual(edge_map)

    # 2. Convert original image (numpy array) to PIL image
    contour_img = Image.fromarray(original_image.copy())
    draw = ImageDraw.Draw(contour_img)

    # 3. Iterate through contours
    for i, contour in enumerate(contours):

        # Compute properties manually
        area = compute_area(contour)
        perimeter = compute_perimeter(contour)
        cx, cy = compute_centroid(contour)

        # Draw centroid (small red dot)
        draw.ellipse((cy-2, cx-2, cy+2, cx+2), fill="red")

        # Draw contour pixels (in green)
        for (x, y) in contour:
            draw.point((y, x), fill="green")

        # Write contour ID near centroid
        draw.text((cy + 5, cx), f"#{i+1}", fill="blue")

        print(f"Contour {i+1}: Area={area:.2f}, Perimeter={perimeter:.2f}, Centroid=({cx},{cy})")

    return np.array(contour_img), len(contours)



# Example usage
if __name__ == "__main__":
    gray = rgb_to_grayscale(image)
    smoothed = gaussian_smoothing(gray, sigma=0.65)
    edges, mag, theta = sobel_edge_detection(smoothed, threshold=100)

    # Step 4: Contour analysis
    contour_img, count = find_and_draw_contours(edges, image)
    print(f"\nTotal contours found: {count}")

    # Show contour image separately using Matplotlib
    plt.figure("Contour Analysis", figsize=(10, 7))
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title("Contour Analysis (Separate View)")
    plt.axis("off")
    plt.show()


    # Display results
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 4, 1); plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.title("Original"); plt.axis('off')
    plt.subplot(1, 4, 2); plt.imshow(gray, cmap='gray'); plt.title("Grayscale"); plt.axis('off')
    plt.subplot(1, 4, 3); plt.imshow(edges, cmap='gray'); plt.title("Edges"); plt.axis('off')
    plt.subplot(1, 4, 4); plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)); plt.title("Contours"); plt.axis('off')
    plt.tight_layout()
    plt.show()