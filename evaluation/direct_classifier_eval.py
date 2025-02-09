import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


def penalty_weighted_accuracy(y_true, y_pred, penalty_matrix):
    total_penalty = 0
    max_penalty = penalty_matrix.max()  # Maximum possible penalty in the matrix
    n = len(y_true)  # Total number of samples

    for i in range(n):
        true_class = y_true[i]
        predicted_class = y_pred[i]
        
        # If misclassified, add the penalty; if correct, penalty is 0
        if true_class != predicted_class:
            total_penalty += penalty_matrix[int(true_class), int(predicted_class)]
    
    # Normalize penalty by the worst-case penalty (n * max_penalty)
    normalized_penalty = total_penalty / (n * max_penalty)
    
    # Calculate PWA
    pwa = 1 - normalized_penalty
    return pwa

penalty_matrix = np.array([
    [0, 100, 100, 100, 100, 100, 10000],    # Good
    [10000, 0, 1, 1, 1, 1, 1000],           # Defect1
    [10000, 1, 0, 1, 1, 1, 1000],           # Defect2
    [10000, 1, 1, 0, 1, 1, 1000],           # Defect3
    [10000, 1, 1, 1, 0, 1, 1000],           # Defect4
    [10000, 1, 1, 1, 1, 0, 1000],           # Defect5
    [10000, 1000, 1000, 1000, 1000, 1000, 0] # Drift
])


# Define a custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 1])
        image = Image.open(img_name).convert("RGB")
        label = self.dataframe.iloc[idx, -1]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load the CSV file
csv_path = '/mnt/disks/location/Y_train.csv'
data = pd.read_csv(csv_path)

# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

path_images = '/mnt/disks/location/input_train'

# Create the datasets
train_dataset = CustomDataset(train_data, path_images, transform=transform)
val_dataset = CustomDataset(val_data, path_images, transform=transform)

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = torch.load('/mnt/disks/location/ens100_sam/saved_models/model.pth', weights_only=False)
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

criterion = nn.CrossEntropyLoss()

val_loss = 0.0
correct, total = 0, 0  # Track accuracy
with torch.no_grad():
    val_loader_tqdm = tqdm(val_loader, desc=f"Final Validation")
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device).long()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * inputs.size(0)

        # Compute accuracy
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels.reshape(-1)).sum().item()
        total += labels.size(0)
            # Live update progress bar
        val_loader_tqdm.set_postfix(loss=loss.item(), accuracy=correct / total)


y_true = labels.reshape(-1).tolist()
y_pred = preds.tolist()

cm = confusion_matrix(y_true, y_pred)


# Print classification report for precision, recall, f1-score
print(classification_report(y_true, y_pred))

val_loss /= len(val_loader.dataset)
val_acc = correct / total
pwa = penalty_weighted_accuracy(labels.reshape(-1).tolist(), preds.tolist(), penalty_matrix)
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, PWAccuracy: {pwa:.4f}')


# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
#plt.show()
plt.savefig('direct_classifier_cm.png')