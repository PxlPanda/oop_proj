import torch
import torch.nn as nn
from torchvision import transforms


import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from .crnn_model import CRNN

class CrnnOcrModel:
    def __init__(self, weights_path, device=None, labels="0123456789АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмпнопрстуфхцчшщъыьэюя"):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = labels
        self.nclass = len(labels) + 1  # +1 for blank

        self.model = CRNN(32, 1, self.nclass, 256).to(self.device)
        checkpoint = torch.load(weights_path, map_location=self.device)
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

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(img)
            preds = log_softmax(preds, dim=2)
            preds = preds.argmax(2).squeeze(1).cpu().numpy()

        char_list = []
        prev = -1
        for idx in preds:
            if idx != prev and idx != self.nclass - 1:
                char_list.append(self.labels[idx])
            prev = idx
        return ''.join(char_list)
