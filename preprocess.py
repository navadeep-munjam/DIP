import cv2
import numpy as np
import pywt
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import io, color, exposure
from skimage.filters import unsharp_mask

def load_image(path, size=None, is_scanner=False):
    """Load and preprocess image based on its source"""
    img = io.imread(path)
    
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        img = color.rgb2gray(img)
    
    # Normalize to [0,1]
    img = img.astype(np.float32)
    if img.max() > 1:
        img /= 255.0
    
    # Scanner-specific preprocessing
    if is_scanner:
        # Enhance contrast for documents
        img = exposure.equalize_adapthist(img, clip_limit=0.03)
        
        # Reduce shadows using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        background = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.divide(img, background)
        
        # Normalize again after processing
        img = np.clip(img, 0, 1)
    
    # Resize if requested
    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    
    return img

def denoise_image(image, is_scanner=False):
    """Apply appropriate denoising based on image source"""
    if is_scanner:
        # Scanner documents often benefit from different processing
        # First pass - wavelet denoising to preserve text edges
        denoised = denoise_wavelet(image, method='BayesShrink', mode='soft', 
                                  wavelet_levels=3, wavelet='sym8')
        
        # Second pass - TV denoising with stronger weight
        denoised = denoise_tv_chambolle(denoised, weight=0.2)
        
        # Sharpening for text clarity
        denoised = unsharp_mask(denoised, radius=1, amount=1.5)
    else:
        # Standard processing for camera images
        denoised = denoise_tv_chambolle(image, weight=0.1)
        denoised = denoise_wavelet(denoised, method='VisuShrink', mode='soft')
    
    return np.clip(denoised, 0, 1)

def create_visualizations(original, denoised, filename=None, is_scanner=False):
    """Create comprehensive visualizations with scanner-specific metrics"""
    plt.figure(figsize=(18, 12))
    original_8bit = (original * 255).astype(np.uint8)
    denoised_8bit = (denoised * 255).astype(np.uint8)
    diff = cv2.absdiff(original_8bit, denoised_8bit)
    
    # 1. Original vs Denoised Comparison
    plt.subplot(2, 3, 1)
    plt.imshow(np.hstack([original_8bit, denoised_8bit]), cmap='gray')
    plt.title('Left: Original | Right: Denoised')
    plt.axis('off')
    
    # 2. Enhanced Difference Visualization
    plt.subplot(2, 3, 2)
    diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    plt.imshow(diff_enhanced, cmap='jet')
    plt.title('Enhanced Noise Removed (Difference)')
    plt.colorbar()
    plt.axis('off')
    
    # 3. Histogram Comparison
    plt.subplot(2, 3, 3)
    plt.hist(original_8bit.ravel(), bins=256, range=(0, 256), 
             color='blue', alpha=0.5, label='Original')
    plt.hist(denoised_8bit.ravel(), bins=256, range=(0, 256), 
             color='red', alpha=0.5, label='Denoised')
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 4. Wavelet Coefficients Visualization (First Level)
    plt.subplot(2, 3, 4)
    wavelet_type = 'db4' if not is_scanner else 'sym8'
    coeffs = pywt.wavedec2(original, wavelet_type, level=3)
    plt.imshow(np.abs(coeffs[0]), cmap='hot')
    plt.title(f'Approx. Coefficients ({wavelet_type})')
    plt.colorbar()
    plt.axis('off')
    
    # 5. Edge Preservation Analysis
    plt.subplot(2, 3, 5)
    threshold1, threshold2 = (50, 100) if is_scanner else (100, 200)
    original_edges = cv2.Canny(original_8bit, threshold1, threshold2)
    denoised_edges = cv2.Canny(denoised_8bit, threshold1, threshold2)
    edge_diff = cv2.bitwise_xor(original_edges, denoised_edges)
    plt.imshow(edge_diff, cmap='gray')
    plt.title('Edge Changes (White = Modified)')
    plt.axis('off')
    
    # 6. Quality Metrics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    psnr = peak_signal_noise_ratio(original_8bit, denoised_8bit)
    ssim = structural_similarity(original_8bit, denoised_8bit)
    
    metrics_text = f"PSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}\n" \
                   f"Noise Removed: {diff.mean():.2f}%\n" \
                   f"Source: {'Scanner' if is_scanner else 'Camera'}"
    
    if is_scanner:
        # Additional metrics for scanner documents
        text_sharpness = cv2.Laplacian(denoised_8bit, cv2.CV_64F).var()
        metrics_text += f"\nText Sharpness: {text_sharpness:.1f}"
    
    plt.text(0.5, 0.5, metrics_text, ha='center', va='center', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.title('Quality Metrics')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=120)
        plt.close()
    else:
        plt.show()

def process_dataset(input_dir, output_dir, vis_dir=None, size=(256, 256), is_scanner=False):
    """Process dataset with source-specific handling"""
    os.makedirs(output_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for filename in tqdm(image_files, desc=f"Processing {'scanner' if is_scanner else 'camera'} images"):
        try:
            input_path = os.path.join(input_dir, filename)
            image = load_image(input_path, size=size, is_scanner=is_scanner)
            denoised = denoise_image(image, is_scanner=is_scanner)
            
            # Save denoised image
            output_path = os.path.join(output_dir, f"denoised_{filename}")
            cv2.imwrite(output_path, (denoised * 255).astype(np.uint8))
            
            # Save comprehensive visualizations
            if vis_dir:
                vis_path = os.path.join(vis_dir, f"analysis_{os.path.splitext(filename)[0]}.png")
                create_visualizations(image, denoised, filename=vis_path, is_scanner=is_scanner)
                
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process camera images
    camera_input = os.path.join(script_dir, "../images/camera")
    camera_output = os.path.join(script_dir, "../images/denoised_camera")
    camera_vis = os.path.join(script_dir, "../images/visualizations_camera")
    process_dataset(camera_input, camera_output, camera_vis, is_scanner=False)
    
    # Process scanner images
    scanner_input = os.path.join(script_dir, "../images/scanner")
    scanner_output = os.path.join(script_dir, "../images/denoised_scanner")
    scanner_vis = os.path.join(script_dir, "../images/visualizations_scanner")
    process_dataset(scanner_input, scanner_output, scanner_vis, is_scanner=True)
    
    print("\nProcessing completed:")
    print(f"Camera visualizations: {camera_vis}")
    print(f"Scanner visualizations: {scanner_vis}")

# import cv2
# import numpy as np
# import pywt
# from skimage.restoration import denoise_tv_chambolle
# import os
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# def create_visualizations(original, denoised, filename=None):
#     """Create comprehensive visualizations for denoising analysis"""
#     plt.figure(figsize=(18, 12))
#     original_8bit = (original * 255).astype(np.uint8)
#     denoised_8bit = (denoised * 255).astype(np.uint8)
#     diff = cv2.absdiff(original_8bit, denoised_8bit)
    
#     # 1. Original vs Denoised Comparison
#     plt.subplot(2, 3, 1)
#     plt.imshow(np.hstack([original_8bit, denoised_8bit]), cmap='gray')
#     plt.title('Left: Original | Right: Denoised')
#     plt.axis('off')
    
#     # 2. Enhanced Difference Visualization
#     plt.subplot(2, 3, 2)
#     diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
#     plt.imshow(diff_enhanced, cmap='jet')
#     plt.title('Enhanced Noise Removed (Difference)')
#     plt.colorbar()
#     plt.axis('off')
    
#     # 3. Histogram Comparison
#     plt.subplot(2, 3, 3)
#     plt.hist(original_8bit.ravel(), bins=256, range=(0, 256), 
#              color='blue', alpha=0.5, label='Original')
#     plt.hist(denoised_8bit.ravel(), bins=256, range=(0, 256), 
#              color='red', alpha=0.5, label='Denoised')
#     plt.title('Pixel Intensity Distribution')
#     plt.xlabel('Intensity Value')
#     plt.ylabel('Frequency')
#     plt.legend()
    
#     # 4. Wavelet Coefficients Visualization (First Level)
#     coeffs = pywt.wavedec2(original, 'db4', level=3)
#     plt.subplot(2, 3, 4)
#     plt.imshow(np.abs(coeffs[0]), cmap='hot')
#     plt.title('Approximation Coefficients (LL)')
#     plt.colorbar()
#     plt.axis('off')
    
#     # 5. Edge Preservation Analysis
#     plt.subplot(2, 3, 5)
#     original_edges = cv2.Canny(original_8bit, 100, 200)
#     denoised_edges = cv2.Canny(denoised_8bit, 100, 200)
#     edge_diff = cv2.bitwise_xor(original_edges, denoised_edges)
#     plt.imshow(edge_diff, cmap='gray')
#     plt.title('Edge Changes (White = Modified)')
#     plt.axis('off')
    
#     # 6. Quality Metrics
#     plt.subplot(2, 3, 6)
#     plt.axis('off')
#     psnr = peak_signal_noise_ratio(original_8bit, denoised_8bit)
#     ssim = structural_similarity(original_8bit, denoised_8bit)
#     metrics_text = f"PSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}\n" \
#                    f"Noise Removed: {diff.mean():.2f}%"
#     plt.text(0.5, 0.5, metrics_text, ha='center', va='center', 
#              fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
#     plt.title('Quality Metrics')
    
#     plt.tight_layout()
    
#     if filename:
#         plt.savefig(filename, bbox_inches='tight', dpi=120)
#         plt.close()
#     else:
#         plt.show()

# def process_dataset(input_dir, output_dir, vis_dir=None, size=(256, 256)):
#     """Enhanced processing with advanced visualizations"""
#     os.makedirs(output_dir, exist_ok=True)
#     if vis_dir:
#         os.makedirs(vis_dir, exist_ok=True)
    
#     image_files = [f for f in os.listdir(input_dir) 
#                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
#     for filename in tqdm(image_files, desc="Processing images"):
#         try:
#             input_path = os.path.join(input_dir, filename)
#             image = load_image(input_path, size=size)
#             denoised = denoise_image(image)
            
#             # Save denoised image
#             output_path = os.path.join(output_dir, f"denoised_{filename}")
#             cv2.imwrite(output_path, (denoised * 255).astype(np.uint8))
            
#             # Save comprehensive visualizations
#             if vis_dir:
#                 vis_path = os.path.join(vis_dir, f"analysis_{os.path.splitext(filename)[0]}.png")
#                 create_visualizations(image, denoised, filename=vis_path)
                
#         except Exception as e:
#             print(f"\nError processing {filename}: {str(e)}")

# if __name__ == "__main__":
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     input_dir = os.path.join(script_dir, "../images/camera")
#     output_dir = os.path.join(script_dir, "../images/denoised_advanced")
#     vis_dir = os.path.join(script_dir, "../images/visualizations_advanced")
    
#     process_dataset(input_dir, output_dir, vis_dir)
#     print(f"\nAdvanced visualizations saved to: {vis_dir}")

# import cv2
# import numpy as np
# import pywt
# from skimage.restoration import denoise_tv_chambolle
# import os
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# def load_image(path, size=(256, 256)):
#     """Load and resize a grayscale image with proper path resolution."""
#     image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise IOError(f"Could not load image at: {path}")
    
#     image = cv2.resize(image, size)
#     return image.astype(np.float32) / 255.0  # Normalize to [0, 1]

# def denoise_image(image, wavelet='db4', level=3, tv_weight=0.1):
#     """Advanced denoising using wavelet shrinkage + TV denoising."""
#     coeffs = pywt.wavedec2(image, wavelet, level=level)
#     threshold = 0.05 * np.max(np.abs(coeffs[-1]))
#     new_coeffs = [coeffs[0]]  # Keep approximation coefficients
#     for detail in coeffs[1:]:
#         new_coeffs.append(tuple(pywt.threshold(c, threshold, mode='soft') for c in detail))
#     reconstructed = pywt.waverec2(new_coeffs, wavelet)
#     return denoise_tv_chambolle(np.clip(reconstructed, 0, 1), weight=tv_weight)

# def plot_comparison(original, denoised, filename=None):
#     """Visualize original vs denoised images with histograms."""
#     plt.figure(figsize=(12, 8))
    
#     # Convert images back to 8-bit for display
#     original_8bit = (original * 255).astype(np.uint8)
#     denoised_8bit = (denoised * 255).astype(np.uint8)
    
#     # Plot images
#     plt.subplot(2, 2, 1)
#     plt.imshow(original_8bit, cmap='gray')
#     plt.title('Original Image')
#     plt.axis('off')
    
#     plt.subplot(2, 2, 2)
#     plt.imshow(denoised_8bit, cmap='gray')
#     plt.title('Denoised Image')
#     plt.axis('off')
    
#     # Plot histograms
#     plt.subplot(2, 2, 3)
#     plt.hist(original_8bit.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
#     plt.title('Original Histogram')
#     plt.xlabel('Pixel Intensity')
#     plt.ylabel('Frequency')
    
#     plt.subplot(2, 2, 4)
#     plt.hist(denoised_8bit.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
#     plt.title('Denoised Histogram')
#     plt.xlabel('Pixel Intensity')
#     plt.ylabel('Frequency')
    
#     plt.tight_layout()
    
#     if filename:
#         plt.savefig(filename, bbox_inches='tight', dpi=100)
#         plt.close()
#     else:
#         plt.show()

# def process_dataset(input_dir, output_dir, vis_dir=None, size=(256, 256)):
#     """
#     Process all images in input_dir, denoise them, and save to output_dir.
    
#     Args:
#         input_dir: Directory containing input images
#         output_dir: Directory to save denoised images
#         vis_dir: Directory to save visualizations (None to skip)
#         size: Target size for resizing images
#     """
#     # Create output directories if they don't exist
#     os.makedirs(output_dir, exist_ok=True)
#     if vis_dir:
#         os.makedirs(vis_dir, exist_ok=True)
    
#     # Get list of image files
#     image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
#     # Process each image with progress bar
#     for filename in tqdm(image_files, desc="Processing images"):
#         try:
#             # Load image
#             input_path = os.path.join(input_dir, filename)
#             image = load_image(input_path, size=size)
            
#             # Denoise image
#             denoised = denoise_image(image)
            
#             # Convert back to 8-bit format for saving
#             denoised_8bit = (denoised * 255).astype(np.uint8)
            
#             # Save denoised image
#             output_path = os.path.join(output_dir, f"denoised_{filename}")
#             cv2.imwrite(output_path, denoised_8bit)
            
#             # Save visualization if requested
#             if vis_dir:
#                 vis_path = os.path.join(vis_dir, f"comparison_{os.path.splitext(filename)[0]}.png")
#                 plot_comparison(image, denoised, filename=vis_path)
            
#         except Exception as e:
#             print(f"\nError processing {filename}: {str(e)}")

# if __name__ == "__main__":
#     # Example usage
#     script_dir = os.path.dirname(os.path.abspath(__file__))
    
#     # Define paths relative to script location
#     input_dir = os.path.join(script_dir, "../images/camera")  # Folder containing original images
#     output_dir = os.path.join(script_dir, "../images/denoised")  # Folder to save denoised images
#     vis_dir = os.path.join(script_dir, "../images/visualizations")  # Folder to save visualizations
    
#     print(f"Input directory: {input_dir}")
#     print(f"Output directory: {output_dir}")
#     print(f"Visualization directory: {vis_dir}")
    
#     # Verify input directory exists
#     if not os.path.exists(input_dir):
#         print(f"\nERROR: Input directory does not exist: {input_dir}")
#         print("\nTroubleshooting:")
#         print("1. Verify the path is correct")
#         print("2. Check if the directory exists")
#         print(f"3. Current working directory: {os.getcwd()}")
#         print(f"4. Directory contents: {os.listdir(os.path.dirname(input_dir))}")
#     else:
#         # Process all images in the input directory
#         process_dataset(input_dir, output_dir, vis_dir)
        
#         print("\nProcessing complete!")
#         print(f"Denoised images saved to: {output_dir}")
#         if vis_dir:
#             print(f"Visualizations saved to: {vis_dir}")
