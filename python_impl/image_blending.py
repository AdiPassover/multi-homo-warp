import numpy as np
import cv2

def image_blending(img1, mask1, img2, mask2, blend_type='linear'):
    """
    Linearly blend two images based on their masks.
    """
    panorama = np.zeros_like(img1)
    
    # Overlap area
    overlap = mask1 & mask2
    
    # Non-overlap areas
    panorama[mask1 & ~overlap] = img1[mask1 & ~overlap]
    panorama[mask2 & ~overlap] = img2[mask2 & ~overlap]
    
    # Linear blending in overlap
    if np.sum(overlap) > 0:
        # Distance transform to find blending weights
        dist1 = cv2.distanceTransform(mask1.astype(np.uint8), cv2.DIST_L2, 5)
        dist2 = cv2.distanceTransform(mask2.astype(np.uint8), cv2.DIST_L2, 5)
        
        weight1 = dist1[overlap]
        weight2 = dist2[overlap]
        
        total_weight = weight1 + weight2 + 1e-8
        alpha1 = weight1 / total_weight
        alpha2 = weight2 / total_weight
        
        alpha1 = np.repeat(alpha1[:, np.newaxis], 3, axis=1)
        alpha2 = np.repeat(alpha2[:, np.newaxis], 3, axis=1)
        
        panorama[overlap] = img1[overlap] * alpha1 + img2[overlap] * alpha2
        
    return panorama
