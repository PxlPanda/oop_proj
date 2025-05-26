# backend/recognition/ocr_dataset.py
from PIL import Image
from torch.utils.data import Dataset
import os

class OcrDataset(Dataset):
    def __init__(self, image_dir, label_file, transform, char_to_idx):
        self.samples = []
        self.image_dir = image_dir
        self.transform = transform
        self.char_to_idx = char_to_idx

        with open(label_file, encoding='utf-8') as f:
            for line in f:
                name, label = line.strip().split(maxsplit=1)
                path = os.path.join(image_dir, name + ".jpg")
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label
