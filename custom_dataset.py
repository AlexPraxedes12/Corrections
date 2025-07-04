"""Custom dataset utilities for medical image fine-tuning."""

from typing import Callable, Optional

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile

# Allow loading of truncated/corrupted images instead of raising an error
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import os
import torch
import warnings

class RFMiDDataset(Dataset):
    """Dataset for multi-label classification of RFMiD retinal images.

    Parameters
    ----------
    image_dir : str
        Directory containing the image files.
    csv_path : str
        Path to the CSV file with image ids and labels.
    transform : callable, optional
        Optional transform to be applied on an image.
    """

    def __init__(self, image_dir: str, csv_path: str,
                 transform: Optional[Callable] = None) -> None:
        self.image_dir = image_dir
        # Read the labels once during initialisation
        self.labels_df = pd.read_csv(csv_path)
        # Columns containing the multi-label targets
        self.label_columns = self.labels_df.columns[3:]
        self.transform = transform
        # Size to be used when an image fails to load
        self._fallback_size = (256, 256)

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.labels_df)

    def __getitem__(self, idx: int):
        """Return the image and label tensor at ``idx``.

        Any failure when loading an image results in a warning and a
        placeholder black image to keep DataLoader behaviour consistent.
        """
        if torch.is_tensor(idx):
            idx = idx.item()

        row = self.labels_df.iloc[idx]
        img_id = f"{row['ID']}.png"
        img_path = os.path.join(self.image_dir, img_id)

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Could not load image '{img_path}': {exc}; using blank image instead.")
            image = Image.new("RGB", self._fallback_size)

        # Retrieve labels and coerce to float32. Missing values become 0.
        labels = (
            row[self.label_columns]
            .fillna(0)
            .astype("float32")
            .to_numpy()
        )

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        labels_tensor = torch.from_numpy(labels)

        return image, labels_tensor
