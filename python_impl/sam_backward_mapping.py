import numpy as np
import cv2

def sam_backward_mapping(img1, img2, sam_H, labels, overlapped_C):
    """
    Warp segments backward using the assigned homographies and blend them.
    """
    sz1, sz2 = img1.shape[:2]
    num_H = sam_H.shape[0]
    
    # Calculate bounding box
    box_xy1 = np.zeros((num_H, 4))
    sam_inv_H = np.zeros((num_H, 9))
    
    corners = np.array([[0, 0, 1], [sz2, 0, 1], [sz2, sz1, 1], [0, sz1, 1]]).T
    
    min_x, min_y = 0, 0
    max_x, max_y = sz2, sz1
    
    for i in range(num_H):
        H = sam_H[i].reshape(3, 3)
        try:
            inv_H = np.linalg.inv(H)
        except:
            inv_H = np.linalg.pinv(H)
        sam_inv_H[i] = inv_H.flatten()
        
        # Warp corners to find bounds
        w_corners = H @ corners
        w_corners = w_corners[:2, :] / (w_corners[2, :] + 1e-8)
        
        box_xy1[i] = [np.floor(w_corners[0].min()), np.floor(w_corners[1].min()),
                      np.ceil(w_corners[0].max()), np.ceil(w_corners[1].max())]
                      
        min_x = min(min_x, box_xy1[i, 0])
        min_y = min(min_y, box_xy1[i, 1])
        max_x = max(max_x, box_xy1[i, 2])
        max_y = max(max_y, box_xy1[i, 3])
        
    offset_x = 2 - min(1, min_x)
    offset_y = 2 - min(1, min_y)
    cw = int(max_x + offset_x)
    ch = int(max_y + offset_y)
    
    if cw * ch > 1e8 or cw <= 0 or ch <= 0:
        return img1, img1, img2, 0, 0
        
    Translation = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)
    
    warped_img1 = np.zeros((ch, cw, 3), dtype=np.float64)
    warped_img2 = np.zeros((ch, cw, 3), dtype=np.float64)
    mask1 = np.zeros((ch, cw), dtype=bool)
    mask2 = np.zeros((ch, cw), dtype=bool)
    
    # Warp img2 globally with Identity? No, img2 is the reference.
    warped_img2_tmp = cv2.warpPerspective(img2, Translation, (cw, ch))
    warped_img2 = warped_img2_tmp.astype(np.float64)
    mask2_tmp = cv2.warpPerspective(np.ones((sz1, sz2), dtype=np.uint8), Translation, (cw, ch))
    mask2 = mask2_tmp > 0
    
    # Warp img1 piece by piece
    for i in range(num_H):
        idx = i + 1
        mask_i = (labels == idx).astype(np.uint8)
        if np.sum(mask_i) == 0:
            continue
            
        img1_masked = img1.copy()
        img1_masked[mask_i == 0] = 0
        
        H = sam_H[i].reshape(3, 3)
        w_img1 = cv2.warpPerspective(img1_masked, Translation @ H, (cw, ch))
        w_mask = cv2.warpPerspective(mask_i, Translation @ H, (cw, ch))
        
        valid = w_mask > 0
        warped_img1[valid] = w_img1[valid]
        mask1[valid] = True
        
    from image_blending import image_blending
    panorama = image_blending(warped_img1, mask1, warped_img2, mask2, 'linear')
    
    return panorama, warped_img1, warped_img2, 0, 0
