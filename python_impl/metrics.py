import numpy as np
import cv2
import torch
import lpips
import skimage.metrics

_lpips_model = None

def get_lpips_model(device):
    global _lpips_model
    if _lpips_model is None:
        # Load AlexNet-based LPIPS model and set it to evaluation mode
        _lpips_model = lpips.LPIPS(net='alex').to(device)
        _lpips_model.eval()
    return _lpips_model

def calculate_psnr(img1, img2, mask):
    """
    Calculate PSNR of img1 and img2 only on the pixels specified by the mask.
    img1, img2: np.ndarray, shape (H, W, 3) or (H, W) in range [0, 1]
    mask: np.ndarray, boolean mask of shape (H, W)
    """
    if not np.any(mask):
        return 0.0
    
    # Calculate MSE on mask pixels
    mse = np.mean((img1[mask] - img2[mask]) ** 2)
    if mse == 0:
        return 100.0
    
    # Since range is [0, 1], max_val is 1.0
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def calculate_ssim(img1, img2, mask):
    """
    Calculate SSIM of img1 and img2 only on the pixels specified by the mask.
    img1, img2: np.ndarray, shape (H, W, 3) or (H, W) in range [0, 1]
    mask: np.ndarray, boolean mask of shape (H, W)
    """
    if not np.any(mask):
        return 0.0
        
    # We find the bounding box of the mask to crop and speed up skimage ssim computation
    y_indices, x_indices = np.where(mask)
    ymin, ymax = y_indices.min(), y_indices.max() + 1
    xmin, xmax = x_indices.min(), x_indices.max() + 1
    
    cropped_img1 = img1[ymin:ymax, xmin:xmax]
    cropped_img2 = img2[ymin:ymax, xmin:xmax]
    cropped_mask = mask[ymin:ymax, xmin:xmax]
    
    # Compute full SSIM map
    try:
        if len(cropped_img1.shape) == 3:
            # For RGB images
            _, ssim_map = skimage.metrics.structural_similarity(
                cropped_img1, cropped_img2, full=True, channel_axis=2, data_range=1.0
            )
        else:
            # For grayscale images
            _, ssim_map = skimage.metrics.structural_similarity(
                cropped_img1, cropped_img2, full=True, data_range=1.0
            )
    except TypeError:
        # Fallback for older scikit-image versions that use multichannel
        if len(cropped_img1.shape) == 3:
            _, ssim_map = skimage.metrics.structural_similarity(
                cropped_img1, cropped_img2, full=True, multichannel=True, data_range=1.0
            )
        else:
            _, ssim_map = skimage.metrics.structural_similarity(
                cropped_img1, cropped_img2, full=True, data_range=1.0
            )
            
    # SSIM on masked region is the mean of SSIM map on mask
    ssim_val = np.mean(ssim_map[cropped_mask])
    return ssim_val

def calculate_lpips(img1, img2, mask):
    """
    Calculate LPIPS of img1 and img2 on the cropped bounding box of the mask,
    with non-overlapping pixels masked to black (0).
    img1, img2: np.ndarray, shape (H, W, 3) in range [0, 1]
    mask: np.ndarray, boolean mask of shape (H, W)
    """
    if not np.any(mask):
        return 0.0
        
    # We find the bounding box of the mask
    y_indices, x_indices = np.where(mask)
    ymin, ymax = y_indices.min(), y_indices.max() + 1
    xmin, xmax = x_indices.min(), x_indices.max() + 1
    
    cropped_img1 = img1[ymin:ymax, xmin:xmax].copy()
    cropped_img2 = img2[ymin:ymax, xmin:xmax].copy()
    cropped_mask = mask[ymin:ymax, xmin:xmax]
    
    # Mask out non-overlapping pixels to zero (black)
    cropped_img1[~cropped_mask] = 0
    cropped_img2[~cropped_mask] = 0
    
    h, w = cropped_mask.shape
    if h < 16 or w < 16:
        # If too small, pad it to avoid deep networks throwing shape errors
        pad_h = max(0, 16 - h)
        pad_w = max(0, 16 - w)
        cropped_img1 = np.pad(cropped_img1, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        cropped_img2 = np.pad(cropped_img2, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    
    # Convert to torch tensor, map [0, 1] to [-1, 1]
    t1 = torch.from_numpy(cropped_img1 * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0).float()
    t2 = torch.from_numpy(cropped_img2 * 2.0 - 1.0).permute(2, 0, 1).unsqueeze(0).float()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t1 = t1.to(device)
    t2 = t2.to(device)
    
    model = get_lpips_model(device)
    
    with torch.no_grad():
        lpips_val = model(t1, t2).item()
        
    return lpips_val

def calculate_all_metrics(img1, img2, mask):
    """
    Calculate PSNR, SSIM, and LPIPS for the overlapping region.
    """
    # Ensure images have 3 channels for LPIPS compatibility
    if len(img1.shape) == 2:
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    else:
        img1_rgb = img1
        img2_rgb = img2
        
    psnr = calculate_psnr(img1_rgb, img2_rgb, mask)
    ssim = calculate_ssim(img1_rgb, img2_rgb, mask)
    lpips_val = calculate_lpips(img1_rgb, img2_rgb, mask)
    
    return ssim, psnr, lpips_val
