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

DATASET_DIR = "dataset"
EPOCHS = 20
BATCH_SIZE = 16
NORMALIZE_CASE = False
MAX_SAMPLES = 1000

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

samples = []

for letter_dir in sorted(os.listdir(DATASET_DIR)):
    abs_path = os.path.join(DATASET_DIR, letter_dir)
    if not os.path.isdir(abs_path):
        continue
    if len(letter_dir) != 1 or not letter_dir.isalpha():
        continue
    for fname in os.listdir(abs_path):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            label = letter_dir
            if NORMALIZE_CASE:
                label = label.lower()
            samples.append((os.path.join(abs_path, fname), label))

samples += load_tsv(os.path.join(DATASET_DIR, "train.tsv"), os.path.join(DATASET_DIR, "train"))
samples += load_tsv(os.path.join(DATASET_DIR, "test.tsv"), os.path.join(DATASET_DIR, "test"))

if MAX_SAMPLES:
    samples = samples[:MAX_SAMPLES]

print(f"📦 Total samples: {len(samples)}")

alphabet = sorted(set(c for _, label in samples for c in label))
print(f"🔤 Alphabet: {''.join(alphabet)}")

weights_dir = "backend/recognition/weights"
os.makedirs(weights_dir, exist_ok=True)
alphabet_path = os.path.join(weights_dir, "alphabet.txt")
with open(alphabet_path, "w", encoding="utf-8") as f:
    f.write("".join(alphabet))

char_to_idx = {c: i for i, c in enumerate(alphabet)}
nclass = len(alphabet) + 1
print(f"🔠 nclass (+CTC blank): {nclass}")

transform = transforms.Compose([
    transforms.Resize((32, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CRNN(32, 1, nclass, 256).to(device)
criterion = nn.CTCLoss(blank=nclass - 1, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Начало обучения:")
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
            print(f"📚 Epoch {epoch} — Batch {i}/{len(loader)} — Loss: {loss.item():.4f}")
    print(f"✅ Epoch {epoch} — Total Loss: {total_loss:.4f} — ⏱ {time.time() - start:.2f}s")

checkpoint_path = os.path.join(weights_dir, "crnn_weights.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'alphabet': ''.join(alphabet)
}, checkpoint_path)
print(f"💾 Модель и алфавит сохранены в: {checkpoint_path}")
