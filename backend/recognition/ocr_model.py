import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .crnn_model import CRNN

class CrnnOcrModel:
    def __init__(self, weights_path, device=None, alphabet_path=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if alphabet_path and os.path.exists(alphabet_path):
            with open(alphabet_path, encoding="utf-8") as f:
                self.labels = f.read().strip()
        else:
            raise ValueError("❌ Alphabet file not found.")

        self.nclass = len(self.labels) + 1
        self.model = CRNN(32, 1, self.nclass, 256).to(self.device)
        checkpoint = torch.load(weights_path, map_location=self.device)

        if "alphabet" in checkpoint:
            saved_alphabet = checkpoint["alphabet"]
            if saved_alphabet != self.labels:
                raise ValueError("❌ Алфавит в checkpoint не совпадает с текущим!")

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 100)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict(self, img):
        from torch.nn.functional import log_softmax
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise ValueError(f"Не удалось загрузить: {img}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img, Image.Image):
            img = np.array(img)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(img)
            preds = log_softmax(preds, dim=2)
            preds = preds.argmax(2).squeeze(1).cpu().numpy()

        result = []
        prev = -1
        for idx in preds:
            if idx != prev and idx != self.nclass - 1:
                result.append(self.labels[idx])
            prev = idx
        return ''.join(result)
