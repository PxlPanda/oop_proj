import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from recognition.crnn_model import CRNN

alphabet = ''.join(sorted(os.listdir('dataset')))
nclass = len(alphabet) + 1
char_to_idx = {c: i for i, c in enumerate(alphabet)}

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder('dataset', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CRNN(32, 1, nclass, 256).to(device)
criterion = nn.CTCLoss(blank=nclass - 1, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def encode(labels):
    targets = []
    lengths = []
    for lbl in labels:
        idxs = [char_to_idx[c] for c in lbl]
        targets.extend(idxs)
        lengths.append(len(idxs))
    return torch.tensor(targets), torch.tensor(lengths)

for epoch in range(10):
    model.train()
    total = 0
    for imgs, targets in loader:
        labels = [dataset.classes[t] for t in targets]
        y, y_len = encode(labels)
        imgs = imgs.to(device)
        pred = model(imgs)
        pred_log = torch.nn.functional.log_softmax(pred, dim=2)
        input_len = torch.full(size=(imgs.size(0),), fill_value=pred.size(0), dtype=torch.long)
        loss = criterion(pred_log, y, input_len, y_len)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"Epoch {epoch+1}/10 — Loss: {total:.4f}")

os.makedirs("recognition/weights", exist_ok=True)
torch.save({'model_state_dict': model.state_dict()}, 'recognition/weights/crnn_weights.pth')
print("✅ Модель обучена и сохранена.")
