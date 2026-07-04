import cv2
import numpy as np

def sift_match(img1, img2):
    """
    SIFT keypoint detection and matching.
    Expects float64 images in [0, 1] range or uint8.
    """
    if img1.dtype == np.float64 or img1.dtype == np.float32:
        img1_uint8 = (img1 * 255).astype(np.uint8)
    else:
        img1_uint8 = img1
        
    if img2.dtype == np.float64 or img2.dtype == np.float32:
        img2_uint8 = (img2 * 255).astype(np.uint8)
    else:
        img2_uint8 = img2

    if len(img1_uint8.shape) == 3:
        gray1 = cv2.cvtColor(img1_uint8, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1_uint8
        
    if len(img2_uint8.shape) == 3:
        gray2 = cv2.cvtColor(img2_uint8, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2_uint8

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Convert to 2xN numpy arrays
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).T
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).T

    # Remove duplicates
    if pts1.size > 0:
        _, unique_indices = np.unique(pts1, axis=1, return_index=True)
        pts1 = pts1[:, unique_indices]
        pts2 = pts2[:, unique_indices]

    return pts1, pts2
