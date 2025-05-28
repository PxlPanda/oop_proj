import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from recognition.crnn_model import CRNN
from recognition.utils_ctc import collate_fn

# === Настройки
DATASET_DIR = "dataset"
EPOCHS = 1
BATCH_SIZE = 16
NORMALIZE_CASE = False
MAX_SAMPLES = 5000  # None для всех примеров

# === Загрузка данных из TSV
def load_tsv(tsv_path, image_folder):
    samples = []
    if not os.path.exists(tsv_path):
        return samples
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            filename, label = parts
            if NORMALIZE_CASE:
                label = label.lower()
            img_path = os.path.join(image_folder, filename)
            if os.path.exists(img_path):
                samples.append((img_path, label))
    return samples

# === Сбор всех сэмплов
samples = []

# 1) одиночные символы
for letter_dir in sorted(os.listdir(DATASET_DIR)):
    abs_letter_path = os.path.join(DATASET_DIR, letter_dir)
    if not os.path.isdir(abs_letter_path):
        continue
    if len(letter_dir) != 1 or not letter_dir.isalpha():
        continue
    for fname in os.listdir(abs_letter_path):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            label = letter_dir
            if NORMALIZE_CASE:
                label = label.lower()
            samples.append((os.path.join(abs_letter_path, fname), label))

# 2) слова из TSV
samples += load_tsv(os.path.join(DATASET_DIR, "train.tsv"), os.path.join(DATASET_DIR, "train"))
samples += load_tsv(os.path.join(DATASET_DIR, "test.tsv"), os.path.join(DATASET_DIR, "test"))

# Ограничение выборки
if MAX_SAMPLES:
    samples = samples[:MAX_SAMPLES]

# === Алфавит и логирование
# Debug: количество сэмплов
print(f"📦 Total samples collected: {len(samples)}")

alphabet = sorted(set(c for _, label in samples for c in label))
alphabet = sorted(set(c for _, label in samples for c in label))
print(f"🔤 Alphabet used for training: {''.join(alphabet)}")
print(f"🔤 Number of characters in alphabet: {len(alphabet)}")

# Сохраняем alphabet.txt вместе с моделью
weights_dir = "backend/recognition/weights"
os.makedirs(weights_dir, exist_ok=True)
alphabet_path = os.path.join(weights_dir, "alphabet.txt")
with open(alphabet_path, "w", encoding="utf-8") as f:
    f.write("".join(alphabet))
print(f"✅ Saved alphabet to: {alphabet_path}")

char_to_idx = {c: i for i, c in enumerate(alphabet)}
nclass = len(alphabet) + 1
print(f"🔠 Computed nclass (alphabet + blank): {nclass}")

# === Трансформации
transform = transforms.Compose([
    transforms.Resize((32, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Dataset и DataLoader
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

# === Модель и оптимизатор
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CRNN(32, 1, nclass, 256).to(device)
criterion = nn.CTCLoss(blank=nclass - 1, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Обучение (всегда выполняется, перезаписывая старые веса)
print("🚀 Starting training (old weights will be overwritten)")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    start = time.time()
    for i, (images, targets, lengths, labels) in enumerate(loader, 1):
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
        if i % 20 == 0:
            print(f"Epoch {epoch} Batch {i}/{len(loader)} — Loss: {loss.item():.4f}")
    end = time.time()
    print(f"Epoch {epoch}/{EPOCHS} — Total Loss: {total_loss:.4f} — ⏱ {end - start:.2f}s")

# === Сохранение модели и верификация
checkpoint_path = os.path.join(weights_dir, "crnn_weights.pth")
# Сохраняем не только веса, но и алфавит для проверки
torch.save({
    'model_state_dict': model.state_dict(),
    'alphabet': ''.join(alphabet)
}, checkpoint_path)
print(f"💾 Saved model weights and alphabet to: {checkpoint_path}")

# Загрузка сразу для верификации
checkpoint = torch.load(checkpoint_path, map_location=device)
saved_alphabet = checkpoint.get('alphabet', None)
if saved_alphabet is None:
    raise KeyError("❌ В checkpoint отсутствует ключ 'alphabet'.")
print(f"🔤 Alphabet saved in checkpoint: {saved_alphabet}")

# Проверяем, что embedding соответствует nclass
nclass_from_weights = checkpoint['model_state_dict']['rnn.1.embedding.weight'].shape[0]
print(f"🧠 Weights expect nclass (including blank): {nclass_from_weights}")
assert nclass_from_weights == nclass, \
    f"Mismatch: nclass defined={nclass}, weights expect={nclass_from_weights}" + \
    " — возможно, alphabet не соответствует весам."
print("🎉 Training complete and verified.")
