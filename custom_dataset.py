from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torch

class RFMiDDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_id = str(row['ID']) + '.png'
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert("RGB")

        # Etiquetas: columnas desde 'ARMD' en adelante (o desde la 3era columna en adelante si es más estable)
        label_columns = self.labels_df.columns[3:]
        labels = torch.tensor(row[label_columns].values.astype('float32'))




        if self.transform:
            image = self.transform(image)

        return image, labels
