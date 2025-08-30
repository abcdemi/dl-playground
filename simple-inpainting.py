import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Model Architecture (Convolutional Autoencoder)
class InpaintingAutoencoder(nn.Module):
    def __init__(self):
        super(InpaintingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # -> [16, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> [16, 16, 16]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # -> [32, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> [32, 8, 8]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), # -> [16, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2), # -> [3, 32, 32]
            nn.Sigmoid() # Use Sigmoid for pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 2. Data Preprocessing and Loading
def create_mask(img):
    """Creates a random mask for an image tensor."""
    mask = torch.clone(img)
    # Create a random binary mask
    random_mask = torch.randint(0, 2, (img.size(0), img.size(1), img.size(2))).bool().to(device)
    # Apply the mask (set pixels to 0)
    mask[random_mask] = 0
    return mask

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(), # Converts PIL Image or numpy.ndarray to tensor and scales to [0, 1]
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 3. Initialize Model, Loss Function, and Optimizer
model = InpaintingAutoencoder().to(device)
criterion = nn.MSELoss() # Mean Squared Error is a good choice for image reconstruction
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Training Loop
num_epochs = 20 # For demonstration; more epochs will yield better results
for epoch in range(num_epochs):
    for data in train_loader:
        original_imgs, _ = data
        original_imgs = original_imgs.to(device)
        
        # Create masked images
        masked_imgs = create_mask(original_imgs)
        
        # Forward pass
        outputs = model(masked_imgs)
        loss = criterion(outputs, original_imgs)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Finished Training")

# 5. Testing and Visualization
def imshow(img_tensor, title=None):
    """Helper function to display a tensor as an image."""
    img = img_tensor.cpu().numpy().transpose((1, 2, 0))
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis('off')

# Get a batch of test images
dataiter = iter(test_loader)
images, _ = next(dataiter)
images = images.to(device)

# Create masks and get model predictions
masked_test_imgs = create_mask(images)
inpainted_imgs = model(masked_test_imgs)

# Move tensors to CPU for visualization
images = images.cpu()
masked_test_imgs = masked_test_imgs.cpu()
inpainted_imgs = inpainted_imgs.detach().cpu()

# Plot the results
plt.figure(figsize=(20, 6))
for i in range(10):
    # Display original
    ax = plt.subplot(3, 10, i + 1)
    imshow(images[i], "Original")

    # Display masked input
    ax = plt.subplot(3, 10, i + 1 + 10)
    imshow(masked_test_imgs[i], "Masked")

    # Display reconstruction
    ax = plt.subplot(3, 10, i + 1 + 20)
    imshow(inpainted_imgs[i], "Inpainted")
plt.show()