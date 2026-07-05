import os
import cv2
import numpy as np
import torch
from enum import Enum


class SegmentationModel(Enum):
    SAM2 = ("SAM-2", True)
    FAST_SAM = ("Fast-SAM", True)
    MOBILE_SAM = ("Mobile-SAM", True)
    YOLOV8 = ("yolov8-seg", True)
    YOLO11 = ("yolo11-seg", True)
    FELZENSZWALB = ("felzenszwalb", False)
    SLIC = ("slic", False)


    def __init__(self, model_name, is_deep):
        self.model_name = model_name
        self.is_deep = is_deep

    def __str__(self):
        return self.model_name

    @staticmethod
    def from_str(str):
        match str.lower():
            case 'sam-2':
                return SegmentationModel.SAM2
            case 'fast-sam':
                return SegmentationModel.FAST_SAM
            case 'mobile-sam':
                return SegmentationModel.MOBILE_SAM
            case 'yolov8-seg':
                return SegmentationModel.YOLOV8
            case 'yolo11-seg':
                return SegmentationModel.YOLO11
            case 'felzenszwalb':
                return SegmentationModel.FELZENSZWALB
            case 'slic':
                return SegmentationModel.SLIC
            case _:
                raise ValueError(f"Unknown model choice: {str}. Valid options are: sam-2, fast-sam, mobile-sam, yolov8-seg, yolo11-seg, felzenszwalb, slic.")



def run_segmentation(image_path: str, output_path: str, model_choice: SegmentationModel, **kwargs) -> np.ndarray:
    """
    Perform image segmentation and save the segmentation labels to a PNG file.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image.
    output_path : str
        Path to save the resulting single-channel segmentation label PNG.
    model_choice : str
        Choice of model: 'sam2', 'fast-sam', 'mobile-sam', 'yolov8-seg', 'yolo11-seg', 'felzenszwalb', 'slic'.
    **kwargs :
        Additional arguments to pass to the models/algorithms.
        - 'checkpoint': Custom weight path for deep learning models.
        - 'device': Target device (e.g., 'cpu', 'cuda').
        - 'conf': Confidence threshold (default 0.25).
        - 'iou': IoU threshold (default 0.7).
        - 'retina_masks': High resolution masks for ultralytics models (default True).
        - 'scale', 'sigma', 'min_size': Arguments for Felzenszwalb segmentation.
        - 'n_segments', 'compactness', 'sigma': Arguments for SLIC segmentation.
        
    Returns:
    --------
    labels : np.ndarray
        2D numpy array of shape (H, W) with integer labels (>= 0).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
        
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    H, W = img.shape[:2]
    
    labels = np.zeros((H, W), dtype=np.int32)
    
    # Deep Learning Models
    from ultralytics import SAM, YOLO, FastSAM

    # Determine device
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Load the appropriate model
    if model_choice.is_deep:
        match model_choice:
            case SegmentationModel.SAM2:
                checkpoint = kwargs.get('checkpoint', 'sam2_t.pt')
                print(f"Loading SAM2 model with checkpoint: {checkpoint} on device: {device}...")
                model = SAM(checkpoint)
            case SegmentationModel.FAST_SAM:
                checkpoint = kwargs.get('checkpoint', 'FastSAM-s.pt')
                print(f"Loading FastSAM model with checkpoint: {checkpoint} on device: {device}...")
                model = FastSAM(checkpoint)
            case SegmentationModel.MOBILE_SAM:
                checkpoint = kwargs.get('checkpoint', 'mobile_sam.pt')
                print(f"Loading Mobile-SAM model with checkpoint: {checkpoint} on device: {device}...")
                model = SAM(checkpoint)
            case SegmentationModel.YOLOV8:
                checkpoint = kwargs.get('checkpoint', 'yolov8n-seg.pt')
                print(f"Loading YOLOv8-seg model with checkpoint: {checkpoint} on device: {device}...")
                model = YOLO(checkpoint)
            case SegmentationModel.YOLO11:
                checkpoint = kwargs.get('checkpoint', 'yolo11n-seg.pt')
                print(f"Loading YOLO11-seg model with checkpoint: {checkpoint} on device: {device}...")
                model = YOLO(checkpoint)
            case _:
                raise ValueError(f"Unknown model choice: {model_choice}. ")

        # Inference arguments
        predict_args = {
            'device': device,
            'conf': kwargs.get('conf', 0.25),
            'iou': kwargs.get('iou', 0.7),
            'retina_masks': kwargs.get('retina_masks', True),
            'verbose': False
        }

        # Run inference
        results = model(image_path, **predict_args)

        if results[0].masks is not None:
            # results[0].masks.data is a tensor of shape (N, H, W)
            masks = results[0].masks.data.cpu().numpy()
            num_masks = masks.shape[0]
            print(f"Generated {num_masks} masks using {model_choice}.")
            for idx, mask in enumerate(masks):
                # Assign unique ID starting from 1 to each mask
                labels[mask > 0.5] = idx + 1
        else:
            print(f"Warning: No masks detected by {model_choice}.")
            
    # Non-Deep Learning Models
    else:
        from skimage.segmentation import felzenszwalb, slic
        
        # Convert image to RGB (skimage algorithms expect RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if model_choice.value == 'felzenszwalb':
            scale = kwargs.get('scale', 100)
            sigma = kwargs.get('sigma', 0.5)
            min_size = kwargs.get('min_size', 50)
            print(f"Running Felzenszwalb segmentation (scale={scale}, sigma={sigma}, min_size={min_size})...")
            
            try:
                raw_labels = felzenszwalb(img_rgb, scale=scale, sigma=sigma, min_size=min_size)
            except (MemoryError, np.exceptions.ArrayMemoryError if hasattr(np, 'exceptions') else MemoryError) as e:
                print(f"Warning: MemoryError ({e}) encountered during Felzenszwalb segmentation at full resolution.")
                print("Retrying with 2x downsampling to conserve memory...")
                h_small, w_small = H // 2, W // 2
                img_small = cv2.resize(img_rgb, (w_small, h_small), interpolation=cv2.INTER_AREA)
                raw_labels_small = felzenszwalb(img_small, scale=scale, sigma=sigma, min_size=min_size)
                raw_labels = cv2.resize(raw_labels_small, (W, H), interpolation=cv2.INTER_NEAREST)
                
            # Ensure label indices start at 1
            labels = raw_labels.astype(np.int32) + 1
            
        print(f"Generated {len(np.unique(labels))} unique segments.")
                         
    # Save the labels to the output path
    max_label = labels.max()
    if max_label < 256:
        labels_save = labels.astype(np.uint8)
    else:
        labels_save = labels.astype(np.uint16)
        
    cv2.imwrite(output_path, labels_save)
    print(f"Successfully saved segmentation map to: {output_path} (dtype: {labels_save.dtype}, max label: {max_label})")
    
    return labels


def batch_segment(folder_path: str, output_folder: str, model_choice: SegmentationModel, suffix: str, **kwargs):
    """
    Batch process all images in a folder for segmentation.

    Parameters:
    -----------
    folder_path : str
        Path to the folder containing input images.
    output_folder : str
        Path to the folder where segmentation label images will be saved.
    model_choice : SegmentationModel
        Choice of segmentation model or algorithm.
    **kwargs :
        Additional arguments to pass to the models/algorithms.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Input folder not found: {folder_path}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(folder_path, filename)
            output_filename = os.path.splitext(filename)[0] + suffix + '.png'
            output_path = os.path.join(output_folder, output_filename)
            print(f"Processing {image_path}...")
            run_segmentation(image_path, output_path, model_choice, **kwargs)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Image Segmentation Module for MHW")
    parser.add_argument('--image', type=str, default=None, help="Path to input image")
    parser.add_argument('--output', type=str, default=None, help="Path to save segmentation label image")
    parser.add_argument('--model', type=str, default='sam-2',
                        choices=['sam-2', 'fast-sam', 'mobile-sam', 'yolov8-seg', 'yolo11-seg', 'felzenszwalb', 'slic'],
                        help="Segmentation model or algorithm choice (default: sam2)")
    parser.add_argument('--folder', type=str, default=None, help="Path to folder for batch processing")
    
    # Deep Learning options
    parser.add_argument('--checkpoint', type=str, default=None, help="Custom model checkpoint weight file")
    parser.add_argument('--device', type=str, default=None, help="Device to run on (e.g. cpu, cuda)")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold for deep learning models")
    parser.add_argument('--iou', type=float, default=0.7, help="IoU threshold for deep learning models")
    
    # Classic options
    # Felzenszwalb parameters
    parser.add_argument('--scale', type=float, default=100.0, help="Felzenszwalb scale parameter")
    parser.add_argument('--sigma', type=float, default=0.5, help="Gaussian smoothing sigma (Felzenszwalb or SLIC)")
    parser.add_argument('--min-size', type=int, default=50, help="Felzenszwalb min component size")
    # SLIC parameters
    parser.add_argument('--n-segments', type=int, default=100, help="SLIC approximate number of segments")
    parser.add_argument('--compactness', type=float, default=10.0, help="SLIC compactness parameter")

    parser.add_argument('--batch-suffix', type=str, default='_segmentation', help="Suffix for output segmentation files in batch mode")
    
    args = parser.parse_args()
    
    # Collect extra arguments
    extra_args = {}
    if args.checkpoint:
        extra_args['checkpoint'] = args.checkpoint
    if args.device:
        extra_args['device'] = args.device
    extra_args['conf'] = args.conf
    extra_args['iou'] = args.iou
    
    extra_args['scale'] = args.scale
    extra_args['sigma'] = args.sigma
    extra_args['min_size'] = args.min_size
    extra_args['n_segments'] = args.n_segments
    extra_args['compactness'] = args.compactness


    if args.image and args.output:
        run_segmentation(args.image, args.output, SegmentationModel.from_str(args.model), **extra_args)
    elif args.folder:
        if not args.output:
            args.output = args.folder
        batch_segment(args.folder, args.output, SegmentationModel.from_str(args.model), args.batch_suffix, **extra_args)
    else:
        raise ValueError("Please provide either --image and --output for single image segmentation, or --folder for batch processing.")