# backend/recognition/train_crnn.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from recognition.crnn_model import CRNN
from recognition.ocr_dataset import OcrDataset
from recognition.utils_ctc import collate_fn

# === Собираем алфавит ===
with open("dataset/words.txt", encoding="utf-8") as f1, open("dataset/sentences.txt", encoding="utf-8") as f2:
    all_text = f1.read() + f2.read()

alphabet = sorted(set(all_text.replace(" ", "")))
char_to_idx = {c: i for i, c in enumerate(alphabet)}
nclass = len(alphabet) + 1  # CTC требует +1 под blank

# === Трансформация изображений ===
transform = transforms.Compose([
    transforms.Resize((32, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Загрузка датасетов ===
words_ds = OcrDataset("dataset/words", "dataset/words.txt", transform, char_to_idx)
sents_ds = OcrDataset("dataset/sentences", "dataset/sentences.txt", transform, char_to_idx)
dataset = ConcatDataset([words_ds, sents_ds])

loader = DataLoader(dataset, batch_size=16, shuffle=True,
                    collate_fn=lambda b: collate_fn(b, char_to_idx))

# === Инициализация модели ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CRNN(32, 1, nclass, 256).to(device)
criterion = nn.CTCLoss(blank=nclass - 1, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Обучение ===
for epoch in range(10):
    model.train()
    total_loss = 0
    for images, targets, lengths, labels in loader:
        images = images.to(device)
        targets = targets.to(device)
        preds = model(images)
        preds_log = F.log_softmax(preds, dim=2)

        input_lengths = torch.full((images.size(0),), preds.size(0), dtype=torch.long)
        loss = criterion(preds_log, targets, input_lengths, lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/10 — Loss: {total_loss:.4f}")

# === Сохранение весов ===
os.makedirs("recognition/weights", exist_ok=True)
torch.save({'model_state_dict': model.state_dict()}, "recognition/weights/crnn_weights.pth")
print("✅ Обучение завершено, модель сохранена.")
