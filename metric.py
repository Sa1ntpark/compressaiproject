import numpy as np
from skimage.metrics import structural_similarity
from skimage.transform import rescale

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: int = 65535) -> float:
    if img1.shape != img2.shape:
        print(img1.shape)
        print(img2.shape)
        raise ValueError("Input images must have the same dimensions.")

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr

def calculate_ms_ssim(img1: np.ndarray, img2: np.ndarray, scales: int = 5) -> float:
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    ms_ssim = []
    for scale in range(scales):
        img1_rescaled = rescale(img1, scale=0.5 ** scale, mode='reflect', anti_aliasing=True)
        img2_rescaled = rescale(img2, scale=0.5 ** scale, mode='reflect', anti_aliasing=True)

        ssim_value = structural_similarity(img1_rescaled, img2_rescaled, data_range=img1_rescaled.max() - img1_rescaled.min())
        ms_ssim.append(ssim_value)
    
    ms_ssim_value = np.prod(ms_ssim) ** (1 / scales)
    return ms_ssim_value