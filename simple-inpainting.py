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

# --- Setup and Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Inpainting Model and Segmentation Model Definitions ---
class InpaintingAutoencoder(nn.Module):
    def __init__(self):
        super(InpaintingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Load the pre-trained segmentation model once
weights = DeepLabV3_ResNet50_Weights.DEFAULT
segmentation_model = deeplabv3_resnet50(weights=weights).to(device).eval()
preprocess_seg = weights.transforms()

# ==============================================================================
# STEP 1: PRE-COMPUTE AND SAVE ALL MASKS (RUN THIS ONCE)
# ==============================================================================
print("--- Starting Mask Pre-computation ---")
MASK_DIR = './cifar10_masks/'
os.makedirs(MASK_DIR, exist_ok=True)

# Use the original CIFAR-10 dataset for this step
cifar10_for_masks = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

# ----- THE FIX IS HERE -----
# The batch size is reduced to 16 to prevent CUDA Out of Memory errors
# during the memory-intensive segmentation model inference.
mask_gen_loader = DataLoader(cifar10_for_masks, batch_size=16, shuffle=False)

# Loop through the dataset and generate a mask for each image
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
            
            mask = mask.unsqueeze(0).unsqueeze(0)
            downsampled_mask = F.interpolate(mask, size=(32, 32), mode='nearest').squeeze().cpu()
            
            # Save the mask as a PyTorch tensor file
            torch.save(downsampled_mask, f"{MASK_DIR}/mask_{image_index}.pt")
            image_index += 1

print(f"--- Finished. {image_index} masks saved to {MASK_DIR} ---")


# ==============================================================================
# STEP 2: CREATE A CUSTOM DATASET TO LOAD IMAGES AND SAVED MASKS
# ==============================================================================
class InpaintingDataset(Dataset):
    def __init__(self, cifar_dataset, mask_directory):
        self.cifar_dataset = cifar_dataset
        self.mask_dir = mask_directory

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        # Get the original image and label
        image, label = self.cifar_dataset[idx]
        
        # Load the corresponding pre-computed mask
        mask_path = os.path.join(self.mask_dir, f"mask_{idx}.pt")
        mask = torch.load(mask_path)
        
        # Add a channel dimension to the mask for broadcasting
        # The final shape will be [1, 32, 32]
        return image, mask.unsqueeze(0)

# ==============================================================================
# STEP 3: TRAIN USING THE EFFICIENT CUSTOM DATASET
# ==============================================================================
# Load the original dataset again to be used by our custom dataset class
train_data_original = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Create our custom dataset which pairs images with their saved masks
train_dataset = InpaintingDataset(cifar_dataset=train_data_original, mask_directory=MASK_DIR)
# The training loader can use a larger batch size as it's very memory-efficient
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model, loss function, and optimizer
inpainting_model = InpaintingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(inpainting_model.parameters(), lr=1e-3)

print("\n--- Starting Efficient Training ---")
num_epochs = 15
for epoch in range(num_epochs):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    running_loss = 0.0
    for original_imgs, masks in progress_bar:
        original_imgs = original_imgs.to(device)
        masks = masks.to(device)
        
        # Masking is now a simple, fast operation
        masked_imgs = original_imgs * (1 - masks)

        # Forward pass, loss calculation, and optimization
        outputs = inpainting_model(masked_imgs)
        loss = criterion(outputs, original_imgs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Update the progress bar with the current loss
        progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})

print("Finished Training")

# ==============================================================================
# STEP 4: VISUALIZE THE RESULTS ON THE TEST SET
# ==============================================================================
print("\n--- Visualizing Results ---")

# Put the model in evaluation mode
inpainting_model.eval()

# Load the test data
test_data_original = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data_original, batch_size=10, shuffle=True) # We only need a few images

# Get a single batch of test images
original_test_imgs, _ = next(iter(test_loader))
original_test_imgs = original_test_imgs.to(device)

# Generate masks for this test batch on the fly
with torch.no_grad():
    batch_for_seg = preprocess_seg(original_test_imgs)
    output = segmentation_model(batch_for_seg)['out']
    seg_maps = torch.argmax(output, dim=1)
    
    test_masks = []
    for seg_map in seg_maps:
        foreground_ids = torch.unique(seg_map)[torch.unique(seg_map) > 0]
        mask = torch.zeros_like(seg_map, dtype=torch.float32)
        if len(foreground_ids) > 0:
            chosen_id = np.random.choice(foreground_ids.cpu().numpy())
            mask[seg_map == chosen_id] = 1.0
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(32, 32), mode='nearest')
        test_masks.append(mask)
    
    test_masks = torch.cat(test_masks, dim=0).to(device)

# Create masked images and get the model's predictions
masked_test_imgs = original_test_imgs * (1 - test_masks)
with torch.no_grad():
    inpainted_imgs = inpainting_model(masked_test_imgs)

# Move all tensors to the CPU for plotting
original_test_imgs = original_test_imgs.cpu()
test_masks = test_masks.cpu()
masked_test_imgs = masked_test_imgs.cpu()
inpainted_imgs = inpainted_imgs.cpu()

# Plot the images
n = 10  # Number of images to display
plt.figure(figsize=(20, 8))

for i in range(n):
    # --- Helper function for plotting ---
    def plot_image(position, img_tensor, title):
        ax = plt.subplot(4, n, position)
        # Permute from (C, H, W) to (H, W, C) for displaying
        plt.imshow(img_tensor.permute(1, 2, 0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0: # Only set title for the first column
            ax.set_title(title, fontsize=14)

    # Row 1: Original Image
    plot_image(i + 1, original_test_imgs[i], "Original")
    
    # Row 2: Generated Mask
    plot_image(i + 1 + n, test_masks[i], "Mask")
    
    # Row 3: Masked Input
    plot_image(i + 1 + 2 * n, masked_test_imgs[i], "Masked Input")
    
    # Row 4: Inpainted Result
    plot_image(i + 1 + 3 * n, inpainted_imgs[i], "Inpainted Result")

plt.tight_layout()
plt.show()