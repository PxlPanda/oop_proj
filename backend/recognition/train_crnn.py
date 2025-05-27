import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from recognition.crnn_model import CRNN
from recognition.utils_ctc import collate_fn

# === Параметры
DATASET_DIR = "dataset"
EPOCHS = 50
BATCH_SIZE = 16
NORMALIZE_CASE = False  # 👈 False = различаем А и а, True = всё в нижний

# === Собираем samples (path, label)

def load_tsv(tsv_path, image_folder):
    samples = []
    if not os.path.exists(tsv_path):
        return samples
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                print(f"❌ Пропущена строка (len={len(parts)}): {repr(line)}")
                continue
            filename, label = parts

            if NORMALIZE_CASE:
                label = label.lower()
            img_path = os.path.join(image_folder, filename)
            if os.path.exists(img_path):
                samples.append((img_path, label))
    return samples

samples = []

# --- Буквы из папок А/Б/В...
for letter_dir in sorted(os.listdir(DATASET_DIR)):
    abs_letter_path = os.path.join(DATASET_DIR, letter_dir)
    if not os.path.isdir(abs_letter_path):
        continue
    if len(letter_dir) != 1 or not letter_dir.isalpha():
        continue  # пропускаем папки типа words/, symbols/

    for fname in os.listdir(abs_letter_path):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            label = letter_dir
            if NORMALIZE_CASE:
                label = label.lower()
            samples.append((os.path.join(abs_letter_path, fname), label))

# --- Слова из tsv
samples += load_tsv(os.path.join(DATASET_DIR, "train.tsv"), os.path.join(DATASET_DIR, "train"))
samples += load_tsv(os.path.join(DATASET_DIR, "test.tsv"), os.path.join(DATASET_DIR, "test"))

# === Алфавит
alphabet = sorted(set(c for _, lbl in samples for c in lbl))
char_to_idx = {c: i for i, c in enumerate(alphabet)}
nclass = len(alphabet) + 1  # blank для CTC

print(f"Алфавит: {''.join(alphabet)}")
print(f"Примеров: {len(samples)}")

# === Трансформация
transform = transforms.Compose([
    transforms.Resize((32, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Dataset
class OcrDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, label

dataset = OcrDataset(samples, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=lambda b: collate_fn(b, char_to_idx))

# === Модель
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CRNN(32, 1, nclass, 256).to(device)

# === Загрузка весов
weights_dir = "backend/recognition/weights"
weights_path = os.path.join(weights_dir, "crnn_weights.pth")
os.makedirs(weights_dir, exist_ok=True)

if os.path.exists(weights_path):
    print("Загрузка сохранённой модели...")
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Веса подгружены")
else:
    print("Начало обучения")

criterion = nn.CTCLoss(blank=nclass - 1, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Обучение
for epoch in range(EPOCHS):
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

    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss:.4f}")

# === Сохранение
torch.save({'model_state_dict': model.state_dict()}, weights_path)
print("Модель сохранена:", weights_path)

with open(os.path.join(weights_dir, "alphabet.txt"), "w", encoding="utf-8") as f:
    f.write("".join(alphabet))
print("Алфавит сохранён.")
