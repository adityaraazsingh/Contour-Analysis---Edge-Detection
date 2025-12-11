import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- COLOR MAP FOR AREA RANGES -----------------------------
def get_color_for_area(area):
    if area < 500:
        return (255, 0, 0)      # Red
    elif area < 2000:
        return (0, 255, 0)      # Green
    elif area < 5000:
        return (0, 0, 255)      # Blue
    else:
        return (255, 0, 255)    # Magenta


# ---------------------- DRAW CONTOURS + LEGEND --------------------------------
def visualize_contours(edge_map, contours):
    # Convert to RGB for coloring
    color_img = np.dstack([edge_map]*3)

    # For legend info
    legend_data = {}

    for contour in contours:
        area = compute_area(contour)
        color = get_color_for_area(area)

        # Store for legend
        range_name = ""
        if area < 500:
            range_name = "Area 0-500"
        elif area < 2000:
            range_name = "Area 500-2000"
        elif area < 5000:
            range_name = "Area 2000-5000"
        else:
            range_name = "Area >5000"

        legend_data[range_name] = color

        # Color all contour pixels
        for (x, y) in contour:
            color_img[x, y] = color

    # ----------------- DISPLAY IMAGE WITH LEGEND -------------------------
    plt.figure(figsize=(8, 6))
    plt.imshow(color_img)
    plt.title("Contours Shaded by Area Range")
    plt.axis("off")

    # Legend
    handles = []
    labels = []

    for label, col in legend_data.items():
        patch = plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=np.array(col)/255, markersize=12)
        handles.append(patch)
        labels.append(label)

    plt.legend(handles, labels, loc="lower right")
    plt.show()


def get_neighbors(x, y):
    """8-connected neighbours"""
    return [
        (x-1, y-1), (x-1, y), (x-1, y+1),
        (x,   y-1),           (x,   y+1),
        (x+1, y-1), (x+1, y), (x+1, y+1)
    ]

def trace_contour(edge_map, start_x, start_y, visited):
    """Trace boundary using simple BFS"""
    stack = [(start_x, start_y)]
    contour = []

    while stack:
        x, y = stack.pop()
        if (x, y) in visited:
            continue
        visited.add((x, y))

        # ---------- FIX: Only keep boundary pixels ----------
        is_boundary = False
        for nx, ny in get_neighbors(x, y):
            if 0 <= nx < edge_map.shape[0] and 0 <= ny < edge_map.shape[1]:
                if edge_map[nx, ny] == 0:
                    is_boundary = True
                    break

        if is_boundary:
            contour.append((x, y))

        # Continue BFS
        for nx, ny in get_neighbors(x, y):
            if 0 <= nx < edge_map.shape[0] and 0 <= ny < edge_map.shape[1]:
                if edge_map[nx, ny] > 0 and (nx, ny) not in visited:
                    stack.append((nx, ny))

    return contour


def compute_perimeter(contour):
    """Sum of distances between consecutive points"""
    peri = 0
    for i in range(len(contour)):
        x1, y1 = contour[i]
        x2, y2 = contour[(i+1) % len(contour)]
        peri += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return peri


def compute_area(contour):
    """Shoelace formula"""
    area = 0
    for i in range(len(contour)):
        x1, y1 = contour[i]
        x2, y2 = contour[(i+1) % len(contour)]
        area += x1*y2 - x2*y1
    return abs(area) / 2


def compute_centroid(contour):
    """Centroid = average of contour points"""
    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]
    return int(sum(xs)/len(xs)), int(sum(ys)/len(ys))


def find_contours_manual(edge_map):
    """Main wrapper — find all contours and compute features"""
    visited = set()
    contours = []

    for x in range(edge_map.shape[0]):
        for y in range(edge_map.shape[1]):
            if edge_map[x, y] > 0 and (x, y) not in visited:
                contour = trace_contour(edge_map, x, y, visited)

                if len(contour) > 0:   # ignore empty
                    contours.append(contour)

    # Display contour info
    for i, contour in enumerate(contours):
        area = compute_area(contour)
        peri = compute_perimeter(contour)
        cx, cy = compute_centroid(contour)
        print(f"Contour {i+1}: Area={area:.2f}, Perimeter={peri:.2f}, Centroid=({cx},{cy})")

    return contours
