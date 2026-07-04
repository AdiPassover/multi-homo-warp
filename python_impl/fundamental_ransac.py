import cv2
import numpy as np

def fundamental_ransac(pts1, pts2, parameters=None):
    """
    Using fundamental matrix for robust fitting to remove outliers.
    pts1, pts2 are 2xN arrays.
    """
    if parameters is None:
        parameters = {}
        
    # The MATLAB thDist was 0.05 on NORMALIZED coordinates.
    # In OpenCV, we operate on unnormalized pixel coordinates. 
    # A standard threshold is around 3.0 pixels.
    # If the user passed thDist, we might want to scale it or just use a default.
    th_dist = parameters.get('pixel_thDist', 3.0) 

    if pts1.shape[1] < 8:
        return pts1, pts2

    pts1_t = pts1.T
    pts2_t = pts2.T

    # Find fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(pts1_t, pts2_t, cv2.FM_RANSAC, th_dist, 0.99)
    
    if mask is not None:
        mask = mask.ravel().astype(bool)
        matches_1 = pts1[:, mask]
        matches_2 = pts2[:, mask]
    else:
        matches_1 = np.empty((2, 0))
        matches_2 = np.empty((2, 0))

    return matches_1, matches_2
