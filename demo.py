import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image
# image = cv2.imread('a.jpg')
# image = cv2.imread('b.jpg')
# image = cv2.imread('c.jpg')
image = cv2.imread('d.jpg')

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

def sobel_edge_detection(image_smoothed, threshold=50):
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

def find_and_draw_contours(edge_map, original_image):
    """Step 4: Find contours from the edge map and draw them on the original image"""
    contours, hierarchy = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = original_image.copy()

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(contour_img, (cx, cy), 4, (0, 0, 255), -1)
        else:
            cx, cy = 0, 0

        # Draw contour and label
        cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)
        cv2.putText(contour_img, f"#{i+1}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        print(f"Contour {i+1}: Area={area:.2f}, Perimeter={perimeter:.2f}, Centroid=({cx},{cy})")

    return contour_img, len(contours)

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