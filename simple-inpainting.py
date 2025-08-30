import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# --- Setup and Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Inpainting Model Architecture (Same as before) ---
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
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- 2. Pre-trained Segmentation Model for Mask Generation ---
weights = DeepLabV3_ResNet50_Weights.DEFAULT
segmentation_model = deeplabv3_resnet50(weights=weights)
segmentation_model.to(device)
segmentation_model.eval()

preprocess_seg = weights.transforms()

# --- 3. Mask Generation Function ---
def create_object_mask_from_model(batch_of_images):
    batch_for_seg = preprocess_seg(batch_of_images)
    with torch.no_grad():
        output = segmentation_model(batch_for_seg)['out']
    
    seg_maps = torch.argmax(output, dim=1)
    
    batch_masks = []
    for seg_map in seg_maps:
        foreground_ids = torch.unique(seg_map)[torch.unique(seg_map) > 0]
        mask = torch.zeros_like(seg_map, dtype=torch.float32)
        if len(foreground_ids) > 0:
            chosen_id = np.random.choice(foreground_ids.cpu().numpy())
            mask[seg_map == chosen_id] = 1.0
        
        mask = mask.unsqueeze(0).unsqueeze(0)
        downsampled_mask = F.interpolate(mask, size=(32, 32), mode='nearest')
        
        # ***** THE FIX IS HERE *****
        # We append the full [1, 1, 32, 32] tensor instead of squeezing it.
        batch_masks.append(downsampled_mask)
    
    # The concatenated result will now have the correct shape: [batch_size, 1, 32, 32]
    return torch.cat(batch_masks, dim=0).to(device)


# --- 4. Data Loading and Preprocessing ---
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- 5. Initialize Model, Loss, and Optimizer ---
inpainting_model = InpaintingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(inpainting_model.parameters(), lr=1e-3)


# --- 6. Training Loop ---
num_epochs = 15
for epoch in range(num_epochs):
    running_loss = 0.0
    for data in train_loader:
        original_imgs, _ = data
        original_imgs = original_imgs.to(device)

        masks = create_object_mask_from_model(original_imgs)
        masked_imgs = original_imgs * (1 - masks) # This will now work correctly

        outputs = inpainting_model(masked_imgs)
        loss = criterion(outputs, original_imgs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

print("Finished Training")


# --- 7. Testing and Visualization ---
dataiter = iter(test_loader)
images, _ = next(dataiter)
images = images.to(device)

final_masks = create_object_mask_from_model(images)
masked_test_imgs = images * (1 - final_masks)
inpainted_imgs = inpainting_model(masked_test_imgs)

# Detach from graph and move to CPU for plotting
images = images.cpu()
final_masks = final_masks.cpu()
masked_test_imgs = masked_test_imgs.cpu()
inpainted_imgs = inpainted_imgs.detach().cpu()

# Plot the results
n = 10
plt.figure(figsize=(20, 8))

# Define a consistent layout grid
rows, cols = 4, n

for i in range(n):
    # Helper to simplify subplot creation
    def plot_image(index, img_tensor, title):
        plt.subplot(rows, cols, index)
        # Permute from [C, H, W] to [H, W, C] for matplotlib
        plt.imshow(img_tensor.permute(1, 2, 0))
        plt.title(title)
        plt.axis('off')

    plot_image(i + 1, images[i], "Original")
    # The mask is [1, 32, 32], permute makes it [32, 32, 1] which imshow handles
    plot_image(i + 1 + cols, final_masks[i], "Generated Mask")
    plot_image(i + 1 + 2 * cols, masked_test_imgs[i], "Masked Image")
    plot_image(i + 1 + 3 * cols, inpainted_imgs[i], "Inpainted Result")

plt.tight_layout()
plt.show()