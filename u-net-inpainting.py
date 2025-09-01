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
# MODEL ARCHITECTURE: U-NET WITH FEATURE REFINEMENT MODULE (FRM)
# Based on the paper "Feature Refinement to Improve High Resolution Image Inpainting"
# ==============================================================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x):
        return self.maxpool_conv(x)

# --- The Feature Refinement Module (FRM) ---
class FeatureRefinementModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # A simple implementation of the refinement idea
        # Using convolutions to learn to refine features from valid regions
        self.refine_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid() # Use Sigmoid to create an attention-like mask
        )

    def forward(self, features, mask):
        # Downsample the mask to match the feature map size
        mask_downsampled = F.interpolate(mask, size=features.shape[2:], mode='nearest')

        # Refine features based on valid regions (where mask is 0)
        refined = self.refine_convs(features * (1 - mask_downsampled))
        
        # Fuse the refined features for the invalid region with the original valid region
        output = (features * (1 - mask_downsampled)) + (refined * mask_downsampled)
        return output

# --- U-Net Decoder Block with the FRM ---
class UpBlockWithFRM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # The FRM is placed after the skip connection is concatenated
        self.frm = FeatureRefinementModule(in_channels)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, mask):
        x1 = self.up(x1)
        if x1.size() != x2.size():
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x2, x1], dim=1)
        x_refined = self.frm(x, mask) # Apply the refinement
        return self.conv(x_refined)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

# --- The Final U-Net with FRM Architecture ---
class UNetWithFRM(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(UNetWithFRM, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Use the new UpBlockWithFRM in the decoder
        self.up1 = UpBlockWithFRM(512, 256)
        self.up2 = UpBlockWithFRM(256, 128)
        self.up3 = UpBlockWithFRM(128, 64)
        
        self.outc = OutConv(64, n_classes)
        self.final_activation = nn.Sigmoid()

    def forward(self, x, mask): # The model now needs the mask as an input
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Pass the mask to each decoder block
        x = self.up1(x4, x3, mask)
        x = self.up2(x, x2, mask)
        x = self.up3(x, x1, mask)
        
        logits = self.outc(x)
        return self.final_activation(logits)

# ==============================================================================
# DATASET AND MASK GENERATION (NO CHANGES)
# ==============================================================================
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

print("\n--- STEP 1: Starting Mask Pre-computation ---")
# ... (This entire section is identical to the previous script, so it's omitted for brevity)
# ... (In a real run, this code would be here)

# ==============================================================================
# STEP 2: TRAIN THE U-NET WITH FRM MODEL
# ==============================================================================
train_data_original = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_dataset = InpaintingDataset(cifar_dataset=train_data_original, mask_directory=MASK_DIR)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Instantiate the NEW model
inpainting_model = UNetWithFRM(n_channels=3, n_classes=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(inpainting_model.parameters(), lr=1e-3)

print("\n--- STEP 2: Starting Efficient Training with FRM ---")
num_epochs = 15
for epoch in range(num_epochs):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    running_loss = 0.0
    for original_imgs, masks in progress_bar:
        original_imgs, masks = original_imgs.to(device), masks.to(device)
        masked_imgs = original_imgs * (1 - masks)
        
        # --- MODEL FORWARD PASS IS NOW DIFFERENT ---
        # The model requires the mask to be passed explicitly
        outputs = inpainting_model(masked_imgs, masks)
        
        loss = criterion(outputs, original_imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})
print("--- Finished Training ---")


# ==============================================================================
# STEP 3 & 4: VISUALIZATION AND METRICS (WITH MODIFICATIONS)
# ==============================================================================
print("\n--- STEP 3 & 4: Visualizing and Evaluating Results ---")
inpainting_model.eval()
test_data_original = datasets.CIFar10(root='./data', train=False, download=True, transform=transforms.ToTensor())
# ... (The rest of the visualization and evaluation code needs to be updated
#      to pass the mask to the model's forward pass)
# For example:
# inpainted_imgs = inpainting_model(masked_test_imgs, test_masks)