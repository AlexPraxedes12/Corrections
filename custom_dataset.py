"""Custom dataset utilities for medical image fine-tuning."""

from typing import Callable, Optional, Dict

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
    source_dirs : dict[str, str], optional
        Mapping from values in the ``source`` column to base image directories.
        Keys are matched case-insensitively. ``rfmid`` defaults to
        ``image_dir``.
    """

    def __init__(
        self,
        image_dir: str,
        csv_path: str,
        transform: Optional[Callable] = None,
        source_dirs: Optional[Dict[str, str]] = None,
    ) -> None:
        self.image_dir = image_dir
        # base directories for additional datasets
        self.source_dirs: Dict[str, str] = {"rfmid": image_dir}
        if source_dirs:
            # normalise keys to lower case for robustness
            self.source_dirs.update({k.lower(): v for k, v in source_dirs.items()})
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
        source = str(row.get("source", "rfmid")).lower()
        base_dir = self.source_dirs.get(source, self.image_dir)

        img_rel = row.get("image")
        if not isinstance(img_rel, str) or img_rel == "" or pd.isna(img_rel):
            img_rel = f"{row['ID']}.png"
        img_path = os.path.join(base_dir, img_rel)

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
