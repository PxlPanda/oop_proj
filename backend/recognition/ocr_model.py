import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from .crnn_model import CRNN


class CrnnOcrModel:
    def __init__(self, weights_path, device=None, alphabet_path=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Загружаем алфавит
        if alphabet_path and os.path.exists(alphabet_path):
            with open(alphabet_path, encoding="utf-8") as f:
                self.labels = f.read().strip()
        else:
            raise ValueError("❌ Не указан путь к alphabet.txt или файл не найден.")

        self.nclass = len(self.labels) + 1  # +1 для CTC blank

        # Загружаем модель
        self.model = CRNN(32, 1, self.nclass, 256).to(self.device)
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Преобразование изображения
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 100)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict(self, img):
        from torch.nn.functional import log_softmax

        # Обработка пути к изображению
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise ValueError(f"❌ Не удалось загрузить изображение по пути: {img}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Обработка PIL.Image
        elif isinstance(img, Image.Image):
            img = np.array(img)

        # Проверка на корректный тип
        if not isinstance(img, np.ndarray):
            raise TypeError("❌ Ожидался путь к изображению, PIL.Image или numpy.ndarray")

        # Приведение к grayscale, если нужно
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Применяем torchvision-преобразования
        img = self.transform(img).unsqueeze(0).to(self.device)

        # Предсказание
        with torch.no_grad():
            preds = self.model(img)
            preds = log_softmax(preds, dim=2)
            preds = preds.argmax(2).squeeze(1).cpu().numpy()

        # Декодируем CTC-выход
        char_list = []
        prev = -1
        for idx in preds:
            if idx != prev and idx != self.nclass - 1:
                char_list.append(self.labels[idx])
            prev = idx

        return ''.join(char_list)
