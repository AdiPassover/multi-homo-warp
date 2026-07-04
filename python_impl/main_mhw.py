import cv2
import numpy as np
import os
import sys
import time

from sift_match import sift_match
from fundamental_ransac import fundamental_ransac
from multi_homo_gen import multi_homo_generation
from multi_homo_fit import multi_homo_fitting
from sam_mask_labeling import sam_mask_labeling
from sam_backward_mapping import sam_backward_mapping

def main():
    parameters = {
        'thDist': 0.05,
        'pixel_thDist': 3.0,
        'dist': 5.0,
        'lambda': 20,
        'beta': 10,
        'maxdatacost': 1e4,
        'gamma': 2e2,
        'display': False
    }
    
    imgpath = '../Imgs/'
    path1 = os.path.join(imgpath, '3_l.jpg')
    path2 = os.path.join(imgpath, '3_r.jpg')
    
    if not os.path.exists(path1) or not os.path.exists(path2):
        print(f"Error: Images not found at {path1} or {path2}.")
        return

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    
    img1_double = img1.astype(np.float64) / 255.0
    img2_double = img2.astype(np.float64) / 255.0

    start_time = time.perf_counter()

    label_path = os.path.join(imgpath, 'segmentation_3_l.png')
    if os.path.exists(label_path):
        labels1 = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if labels1 is not None and len(labels1.shape) == 3:
            labels1 = labels1[:,:,0] # just need 1 channel
    else:
        labels1 = np.zeros(img1.shape[:2], dtype=np.int32)
        print(f"Warning: Segmentation not found at {label_path}")

    # Step 1: Feature matching
    print("Extracting and matching SIFT features...")
    pts1, pts2 = sift_match(img1_double, img2_double)
    
    # Step 2: Fundamental RANSAC
    print("Running Fundamental Matrix RANSAC...")
    matches_1, matches_2 = fundamental_ransac(pts1, pts2, parameters)
    print(f"Inliers remaining: {matches_1.shape[1]}")
    
    # Step 3: Multi-homography Initialization
    print("Generating Initial Homographies...")
    init_H, _ = multi_homo_generation(matches_1, matches_2)
    
    # Step 4: Multi-homography Fitting (PEaRL)
    print("Running Multi-homography Fitting (Graph-Cut)...")
    multi_homos, cell_matches = multi_homo_fitting(matches_1, matches_2, img1_double, img2_double, labels1, init_H, parameters)
    print(f"Number of homographies: {len(multi_homos)}")
    
    # Step 5: SAM Mask Labeling
    print("Labeling SAM masks...")
    final_labels1, final_homos, overlapped_C = sam_mask_labeling(img1_double, img2_double, labels1, multi_homos, cell_matches)
    
    # Step 6: Backward Mapping and Blending
    print("Warping and Blending...")
    panorama, w_img1, w_img2, ssim, psnr = sam_backward_mapping(img1_double, img2_double, final_homos, final_labels1, overlapped_C)

    end_time = time.perf_counter()
    print(f"Execution time: {(end_time - start_time):.6f} seconds")
    
    cv2.imwrite('panorama.jpg', (panorama * 255).astype(np.uint8))
    print("Saved panorama.jpg")

if __name__ == '__main__':
    main()
