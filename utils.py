import cv2
import numpy as np

def detect_edges(gray, blur, fudgefactor):
    # Normalize the gray image to float32 and subtract the blurred image
    gray_norm = gray / 255.0
    enhanced = cv2.subtract(np.float32(gray_norm), np.float32(blur))
    
    # Apply Sobel operators to detect edges
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(sobelx, sobely)

    # Thresholding for edge detection
    threshold = 4 * fudgefactor * np.mean(mag)
    mag[mag < threshold] = 0
    mag[mag >= threshold] = 255
    edge_mask = mag.astype(np.uint8)

    return edge_mask
