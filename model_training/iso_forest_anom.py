import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

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
        #image.verify()
        label = self.dataframe.iloc[idx, -1]
        if self.transform:
            image = self.transform(image)
        return image, label

model = torch.load('/mnt/disks/location/ens100_sam/saved_models/model.pth', weights_only=False)
print(model)

model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last FC layer for features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load dataset
# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_csv_path = '/mnt/disks/location/Y_train.csv'
df_train = pd.read_csv(train_csv_path)

train_path_images = '/mnt/disks/location/input_train'
dataset = CustomDataset(dataframe=df_train, img_dir=train_path_images, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

print(f"Total images loaded in dataset: {len(dataset)}")

features = []
with torch.no_grad():
    for images, _ in data_loader:
        images = images.to(device)
        outputs = model(images)
        features.append(outputs.view(images.size(0), -1).cpu().numpy())

features = np.vstack(features)

# Train Isolation Forest for anomaly detection
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(features)


#test set anomalies -> load test images, run the pipeline to get test features, then predict with anomaly
test_path_images = '/mnt/disks/location/input_test'
test_csv_path = '/mnt/disks/location/Y_random.csv'
df_test = pd.read_csv(test_csv_path)

test_dataset = CustomDataset(dataframe=df_test, img_dir=test_path_images, transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Total images loaded in dataset: {len(test_dataset)}")

features = []
with torch.no_grad():
    for images, _ in test_data_loader:
        images = images.to(device)
        outputs = model(images)
        features.append(outputs.view(images.size(0), -1).cpu().numpy())

features = np.vstack(features)
print(len(features))

# Predict anomalies (-1 = anomaly, 1 = normal)
preds = iso_forest.predict(features)
anomaly_indices = np.where(preds == -1)[0]
print(f"Detected {len(anomaly_indices)} anomalies out of {len(features)} images.")

test_df = pd.read_csv('/mnt/disks/location/Y_random.csv')
test_df['forest_anom'] = preds.reshape(-1,1)
test_df.to_csv('trial.csv')
print('saved')
print(test_df.forest_anom.value_counts())