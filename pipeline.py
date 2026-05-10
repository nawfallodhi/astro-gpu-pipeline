
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
import numpy as np
import time
import matplotlib.pyplot as plt
import tifffile
from skimage.transform import resize

# ── Load Data ──────────────────────────────────────────────────────────────
print("Loading Tulip Nebula data...")
tif_array = tifffile.imread("TULIP_4HOURS_20MINS.tif").astype(np.float32)
print(f"Raw shape: {tif_array.shape} | Dtype: {tif_array.dtype}")
print(f"Pixel range: {tif_array.min():.4f} to {tif_array.max():.4f}")

# Normalize and resize for processing
tif_array = (tif_array - tif_array.min()) / (tif_array.max() - tif_array.min())
tif_small = resize(tif_array, (480, 720, 3), anti_aliasing=True).astype(np.float32)
print(f"Working shape: {tif_small.shape} | Memory: {tif_small.nbytes/1e6:.1f} MB")

# ── CPU Pipeline ────────────────────────────────────────────────────────────
def cpu_pipeline(image):
    from scipy.ndimage import gaussian_filter as sf
    result = np.zeros_like(image)
    for c in range(3):
        channel = image[:,:,c]
        p0_5, p99_5 = np.percentile(channel, (0.5, 99.5))
        stretched = np.clip((channel - p0_5) / (p99_5 - p0_5), 0, 1)
        stretched = np.power(stretched, 1.3)
        blurred = sf(stretched, sigma=1.5)
        result[:,:,c] = np.clip(stretched + 0.5*(stretched - blurred), 0, 1)
    result = np.stack([
        np.clip(result[:,:,0] * 1.2, 0, 1),
        np.clip(result[:,:,1] * 0.65, 0, 1),
        np.clip(result[:,:,2] * 0.15, 0, 1),
    ], axis=-1)
    return np.clip(result * 1.3, 0, 1)

# ── GPU Pipeline ────────────────────────────────────────────────────────────
def gpu_pipeline(image):
    img_gpu = cp.array(image)
    channels = []
    for c in range(3):
        channel = img_gpu[:,:,c]
        p0_5 = cp.percentile(channel, 0.5)
        p99_5 = cp.percentile(channel, 99.5)
        stretched = cp.clip((channel - p0_5) / (p99_5 - p0_5), 0, 1)
        stretched = cp.power(stretched, 1.3)
        blurred = gaussian_filter(stretched, sigma=1.5)
        sharpened = cp.clip(stretched + 0.5*(stretched - blurred), 0, 1)
        channels.append(sharpened)
    result = cp.stack([
        cp.clip(channels[0] * 1.2, 0, 1),
        cp.clip(channels[1] * 0.65, 0, 1),
        cp.clip(channels[2] * 0.15, 0, 1),
    ], axis=-1)
    result = cp.clip(result * 1.3, 0, 1)
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(result)

# ── Benchmark ───────────────────────────────────────────────────────────────
print("\nWarming up GPU...")
_ = gpu_pipeline(tif_small)

print("Benchmarking CPU...")
start = time.time()
for _ in range(20):
    cpu_result = cpu_pipeline(tif_small)
cpu_time = (time.time() - start) / 20

print("Benchmarking GPU...")
start = time.time()
for _ in range(20):
    gpu_result = gpu_pipeline(tif_small)
gpu_time = (time.time() - start) / 20

speedup = cpu_time / gpu_time
print(f"\nCPU: {cpu_time:.4f}s")
print(f"GPU: {gpu_time:.4f}s")
print(f"Speedup: {speedup:.1f}x")

# ── Visualize ───────────────────────────────────────────────────────────────
h, w = gpu_result.shape[:2]
crop = gpu_result[h//3:2*h//3, w//2:, :]

fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor="black")
axes[0].imshow(tif_small)
axes[0].set_title("Raw Data (4hr 20min exposure)", fontsize=13, color="white")
axes[0].axis("off")
axes[1].imshow(np.clip(gpu_result, 0, 1))
axes[1].set_title(f"GPU Processed | SHO Palette | {speedup:.1f}x faster than CPU", 
                  fontsize=13, color="white")
axes[1].axis("off")
axes[2].imshow(np.clip(crop, 0, 1))
axes[2].set_title("Tulip Nebula - Zoomed", fontsize=13, color="white")
axes[2].axis("off")
plt.suptitle("Tulip Nebula Region - GPU Astrophotography Pipeline", 
             fontsize=15, color="white")
plt.tight_layout()
plt.savefig("tulip_final.png", dpi=200, bbox_inches="tight", facecolor="black")
plt.show()
print(f"\nSaved tulip_final.png")
