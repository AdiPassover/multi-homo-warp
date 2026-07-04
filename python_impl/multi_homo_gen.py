import cv2
import numpy as np

def multi_homo_generation(pts1, pts2):
    """
    Use RANSAC-based method to find the initial models (homographies) 
    for Multi-homography fitting.
    pts1, pts2 are 2xN numpy arrays.
    """
    rest_pts1 = pts1.T
    rest_pts2 = pts2.T
    
    init_H = []
    cell_matches = []
    
    # In MATLAB: while length(rest_pts1)>=50
    while len(rest_pts1) >= 50:
        # threshold is not strictly specified in MATLAB for homoRANSAC, 
        # usually 3 to 5 pixels in unnormalized coordinates.
        H, mask = cv2.findHomography(rest_pts1, rest_pts2, cv2.RANSAC, 5.0)
        
        if H is None or mask is None:
            break
            
        mask = mask.ravel().astype(bool)
        
        inliers1 = rest_pts1[mask]
        inliers2 = rest_pts2[mask]
        
        if len(inliers1) < 4:
            break
            
        # Re-fit homography on all inliers for refinement
        H_ref, _ = cv2.findHomography(inliers1, inliers2, 0)
        if H_ref is not None:
            init_H.append(H_ref)
        else:
            init_H.append(H)
            
        cell_matches.append((inliers1.T, inliers2.T))
        
        # Remove inliers to find the next model
        rest_pts1 = rest_pts1[~mask]
        rest_pts2 = rest_pts2[~mask]
        
    return init_H, cell_matches
