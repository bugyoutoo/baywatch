import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms

class MarineDebrisDataset(Dataset):
    def __init__(self, img_dir, csv_file=None, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        if csv_file is not None:
            self.labels_df = pd.read_csv(csv_file)
            self.filenames = self.labels_df["filename"].tolist()
        else:
            # Just list all .jpg files in the directory
            self.filenames = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
            self.labels_df = None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(img_path).convert("RGB")

        if self.labels_df is not None:
            row = self.labels_df.iloc[idx]
            label = torch.tensor([row["debris"], row["cloud"]], dtype=torch.float32)
        else:
            label = torch.tensor([0, 0], dtype=torch.float32)  # dummy label

        if self.transform:
            image = self.transform(image)

        return image, label