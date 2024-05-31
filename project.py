import torch
import torchvision
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 600
weight_decay = 0.001  # Weight decay for regularisation

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Transformations for better data augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to dimension 224 x 224
        transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(30),  # Randomly rotate the image with range 30
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Randomly change the brightness, contrast, saturation and hue of images
        transforms.ToTensor(),  # Convert the image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisation
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to dimension 224 x 224
        transforms.ToTensor(),  # Convert the image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisation
    ])
}

# Directory of dataset
data_dir = 'dataset_flower102/'

# Load train data
train_dataset = datasets.Flowers102(root=data_dir, split='train', transform=data_transforms['train'], download=True)

# Load test data
test_dataset = datasets.Flowers102(root=data_dir, split='test', transform=data_transforms['test'], download=True)

# Load validation data
valid_dataset = datasets.Flowers102(root=data_dir, split='val', transform=data_transforms['test'], download=True)

# Loader for train, test and valid
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Define the VGG16 model
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(512 * 14 * 14, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(512, 102),
            nn.BatchNorm1d(102)
        )

    def forward(self, x):
        return self.model(x)

model = VGG16().to(device)

# Initialize the model, criterion, and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Training and validation loop
train_losses, valid_losses = [], []
train_accuracies, valid_accuracies = [], []

for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Training step
    for i, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(data)

        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        total += targets.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        if (i + 1) % 8 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}]")

    train_accuracy = 100.0 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.2f}%")
    
    # Step the scheduler
    scheduler.step()

    # Calculate average training loss
    train_losses.append(running_loss / len(train_loader))

    # Set the model to evaluation mode
    model.eval()
    valid_loss = 0
    n_correct = 0
    n_samples = 0
    
    # Validation step
    with torch.no_grad():
        for data, targets in valid_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(data)

            # Calculate loss
            loss = criterion(outputs, targets)
            valid_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == targets).sum().item()
            n_samples += targets.size(0)
    
    valid_loss /= len(valid_loader)
    valid_losses.append(valid_loss)
    valid_accuracy = 100.0 * n_correct / n_samples
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%')

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict()
}, "flowers-102-vgg16.pth")

# Load the model
checkpoint = torch.load("flowers-102-vgg16.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Evaluation on test set
model.eval()
n_correct = 0
n_samples = 0

# Testing step
with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == targets).sum().item()
        n_samples += targets.size(0)

    accuracy = 100.0 * n_correct / n_samples
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
    print(f'{n_correct} / {n_samples} correct images')

'''
# Plot the training and validation losses
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot the training and validation accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()
'''
