# ==============================================================================
# INSTALL DEPENDENCIES
# ==============================================================================
print("Installing necessary libraries: torchmetrics, lpips, torch-fidelity...")
!pip install torchmetrics lpips torch-fidelity -q
print("Installation complete.")

# ==============================================================================
# IMPORTS AND SETUP
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import subprocess
import re
import torchvision.utils

# --- Setup and Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# MODEL AND DATASET DEFINITIONS
# ==============================================================================

# --- Inpainting Model Definition ---
class InpaintingAutoencoder(nn.Module):
    def __init__(self):
        super(InpaintingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- Custom Dataset for Efficient Loading ---
class InpaintingDataset(Dataset):
    def __init__(self, cifar_dataset, mask_directory):
        self.cifar_dataset = cifar_dataset
        self.mask_dir = mask_directory

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        mask_path = os.path.join(self.mask_dir, f"mask_{idx}.pt")
        mask = torch.load(mask_path)
        return image, mask.unsqueeze(0)

# ==============================================================================
# STEP 1: PRE-COMPUTE AND SAVE ALL MASKS (RUNS ONCE)
# ==============================================================================
print("\n--- STEP 1: Starting Mask Pre-computation ---")
MASK_DIR = './cifar10_masks/'
os.makedirs(MASK_DIR, exist_ok=True)

# Load the pre-trained segmentation model
weights = DeepLabV3_ResNet50_Weights.DEFAULT
segmentation_model = deeplabv3_resnet50(weights=weights).to(device).eval()
preprocess_seg = weights.transforms()

cifar10_for_masks = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
mask_gen_loader = DataLoader(cifar10_for_masks, batch_size=16, shuffle=False) # Reduced batch size for memory

with torch.no_grad():
    image_index = 0
    for images, _ in tqdm(mask_gen_loader, desc="Generating and Saving Masks"):
        images = images.to(device)
        batch_for_seg = preprocess_seg(images)
        output = segmentation_model(batch_for_seg)['out']
        seg_maps = torch.argmax(output, dim=1)

        for seg_map in seg_maps:
            foreground_ids = torch.unique(seg_map)[torch.unique(seg_map) > 0]
            mask = torch.zeros_like(seg_map, dtype=torch.float32)
            if len(foreground_ids) > 0:
                chosen_id = np.random.choice(foreground_ids.cpu().numpy())
                mask[seg_map == chosen_id] = 1.0
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(32, 32), mode='nearest').squeeze().cpu()
            torch.save(mask, f"{MASK_DIR}/mask_{image_index}.pt")
            image_index += 1
print(f"--- Finished. {image_index} masks saved to {MASK_DIR} ---")

# ==============================================================================
# STEP 2: TRAIN THE INPAINTING MODEL
# ==============================================================================
train_data_original = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_dataset = InpaintingDataset(cifar_dataset=train_data_original, mask_directory=MASK_DIR)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

inpainting_model = InpaintingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(inpainting_model.parameters(), lr=1e-3)

print("\n--- STEP 2: Starting Efficient Training ---")
num_epochs = 15
for epoch in range(num_epochs):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    running_loss = 0.0
    for original_imgs, masks in progress_bar:
        original_imgs, masks = original_imgs.to(device), masks.to(device)
        masked_imgs = original_imgs * (1 - masks)
        outputs = inpainting_model(masked_imgs)
        loss = criterion(outputs, original_imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})
print("--- Finished Training ---")

# ==============================================================================
# STEP 3: VISUALIZE RESULTS ON A TEST BATCH
# ==============================================================================
print("\n--- STEP 3: Visualizing Results ---")
inpainting_model.eval()
test_data_original = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader_sample = DataLoader(test_data_original, batch_size=10, shuffle=True)
original_test_imgs, _ = next(iter(test_loader_sample))
original_test_imgs = original_test_imgs.to(device)

with torch.no_grad():
    output = segmentation_model(preprocess_seg(original_test_imgs))['out']
    seg_maps = torch.argmax(output, dim=1)
    test_masks = []
    for seg_map in seg_maps:
        foreground_ids = torch.unique(seg_map)[torch.unique(seg_map) > 0]
        mask = torch.zeros_like(seg_map, dtype=torch.float32)
        if len(foreground_ids) > 0:
            mask[seg_map == np.random.choice(foreground_ids.cpu().numpy())] = 1.0
        test_masks.append(F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(32, 32), mode='nearest'))
    test_masks = torch.cat(test_masks, dim=0).to(device)
    masked_test_imgs = original_test_imgs * (1 - test_masks)
    inpainted_imgs = inpainting_model(masked_test_imgs)

# Plotting
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    def plot_image(position, img_tensor, title):
        ax = plt.subplot(4, n, position)
        plt.imshow(img_tensor.cpu().permute(1, 2, 0))
        ax.get_xaxis().set_visible(False); ax.get_yaxis().set_visible(False)
        if i == 0: ax.set_title(title, fontsize=14)
    plot_image(i + 1, original_test_imgs[i], "Original"); plot_image(i + 1 + n, test_masks[i], "Mask")
    plot_image(i + 1 + 2 * n, masked_test_imgs[i], "Masked Input"); plot_image(i + 1 + 3 * n, inpainted_imgs[i], "Inpainted Result")
plt.tight_layout(); plt.show()

# ==============================================================================
# STEP 4: CALCULATE ALL METRICS (PSNR, SSIM, LPIPS, FID)
# ==============================================================================
print("\n--- STEP 4: Starting Quantitative Evaluation on Full Test Set ---")
import torchmetrics
import lpips

# --- Initialize Metrics ---
psnr_metric = torchmetrics.PeakSignalNoiseRatio().to(device)
ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure().to(device)
lpips_metric = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

# --- Setup Directories for FID Patches ---
REAL_PATCHES_DIR = '/tmp/real_patches/'; FAKE_PATCHES_DIR = '/tmp/fake_patches/'
os.makedirs(REAL_PATCHES_DIR, exist_ok=True); os.makedirs(FAKE_PATCHES_DIR, exist_ok=True)
!rm -rf {REAL_PATCHES_DIR}/*; !rm -rf {FAKE_PATCHES_DIR}/*

test_loader_full = DataLoader(test_data_original, batch_size=32, shuffle=False)
patch_count = 0

with torch.no_grad():
    for original_imgs, _ in tqdm(test_loader_full, desc="Evaluating Metrics and Saving Patches"):
        original_imgs = original_imgs.to(device)
        
        # --- Generate masks ---
        output = segmentation_model(preprocess_seg(original_imgs))['out']
        seg_maps = torch.argmax(output, dim=1)
        masks = []
        for seg_map in seg_maps:
            foreground_ids = torch.unique(seg_map)[torch.unique(seg_map) > 0]
            mask = torch.zeros_like(seg_map, dtype=torch.float32)
            if len(foreground_ids) > 0:
                mask[seg_map == np.random.choice(foreground_ids.cpu().numpy())] = 1.0
            masks.append(F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(32, 32), mode='nearest'))
        masks = torch.cat(masks, dim=0).to(device)

        # --- Get Inpainted Results ---
        masked_imgs = original_imgs * (1 - masks)
        inpainted_imgs = inpainting_model(masked_imgs)
        
        # --- Update PSNR, SSIM, LPIPS metrics (on full images) ---
        psnr_metric.update(inpainted_imgs, original_imgs)
        ssim_metric.update(inpainted_imgs, original_imgs)
        lpips_metric.update(inpainted_imgs * 2 - 1, original_imgs * 2 - 1) # LPIPS expects [-1, 1]

        # --- Extract and Save Patches for FID ---
        for i in range(original_imgs.size(0)):
            single_mask = masks[i].squeeze()
            if not torch.any(single_mask): continue
            rows, cols = torch.any(single_mask, axis=1), torch.any(single_mask, axis=0)
            rmin, rmax = torch.where(rows)[0][[0, -1]]; cmin, cmax = torch.where(cols)[0][[0, -1]]
            real_patch = original_imgs[i][:, rmin:rmax+1, cmin:cmax+1]
            fake_patch = inpainted_imgs[i][:, rmin:rmax+1, cmin:cmax+1]
            torchvision.utils.save_image(real_patch, os.path.join(REAL_PATCHES_DIR, f'p_{patch_count}.png'))
            torchvision.utils.save_image(fake_patch, os.path.join(FAKE_PATCHES_DIR, f'p_{patch_count}.png'))
            patch_count += 1

# --- Compute Final Scores ---
final_psnr = psnr_metric.compute()
final_ssim = ssim_metric.compute()
final_lpips = lpips_metric.compute()
fid_score = "N/A"

if patch_count > 1:
    print(f"\nSaved {patch_count} patches. Calculating FID score...")
    cmd = ['torch-fidelity', '--gpu', '0', '--fid', '--input1', REAL_PATCHES_DIR, '--input2', FAKE_PATCHES_DIR]
    result = subprocess.run(cmd, capture_output=True, text=True)
    match = re.search(r"frechet_inception_distance:\s*(\d+\.\d+)", result.stdout)
    if match:
        fid_score = float(match.group(1))
    else:
        print("Could not parse FID score. Raw output:")
        print(result.stdout)
        print(result.stderr)
else:
    print("Not enough patches generated to calculate FID.")

# --- Final Report ---
print("\n\n--- FINAL EVALUATION REPORT ---")
print(f"Peak Signal-to-Noise Ratio (PSNR): {final_psnr:.4f} dB (Higher is better)")
print(f"Structural Similarity Index (SSIM): {final_ssim:.4f} (Higher is better)")
print(f"Learned Perceptual Patch Similarity (LPIPS): {final_lpips:.4f} (Lower is better)")
print(f"Fréchet Inception Distance (FID on patches): {fid_score if isinstance(fid_score, str) else f'{fid_score:.4f}'} (Lower is better)")

# --- Clean up ---
!rm -rf {REAL_PATCHES_DIR}; !rm -rf {FAKE_PATCHES_DIR}; !rm -rf {MASK_DIR}
print("\nCleaned up temporary directories.")