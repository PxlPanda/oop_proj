import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from recognition.crnn_model import CRNN
import torch.nn.functional as F

# === Путь к модели и алфавиту ===
WEIGHTS_PATH = "backend/recognition/weights/crnn_weights.pth"
ALPHABET_PATH = "backend/recognition/weights/alphabet.txt"

# === Подготовка алфавита ===
if not os.path.exists(ALPHABET_PATH):
    print("❌ Файл alphabet.txt не найден.")
    sys.exit(1)

with open(ALPHABET_PATH, encoding="utf-8") as f:
    alphabet = list(f.read())

idx_to_char = {i: c for i, c in enumerate(alphabet)}
nclass = len(alphabet) + 1  # +1 blank

# === Загрузка модели ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CRNN(32, 1, nclass, 256).to(device)
checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# === Преобразование изображения
transform = transforms.Compose([
    transforms.Resize((32, 100)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Получаем папку от пользователя
if len(sys.argv) != 2:
    print("🔧 Использование: python3 batch_predict.py dataset/lowercase")
    sys.exit(1)

img_dir = sys.argv[1]
if not os.path.isdir(img_dir):
    print("❌ Папка не найдена:", img_dir)
    sys.exit(1)

# === Перебираем файлы
total = 0
correct = 0

for fname in sorted(os.listdir(img_dir)):
    if not fname.lower().endswith(".jpg"):
        continue

    label = os.path.splitext(fname)[0]  # имя файла — метка
    path = os.path.join(img_dir, fname)
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)
        log_probs = F.log_softmax(preds, dim=2)
        pred_indices = log_probs.argmax(2).squeeze(1).cpu().numpy()

    # CTC-декодирование
    result = []
    prev = -1
    for idx in pred_indices:
        if idx != prev and idx != nclass - 1:
            result.append(idx_to_char.get(idx, "?"))
        prev = idx

    pred_str = ''.join(result)
    status = "✅" if pred_str == label else "❌"
    print(f"{fname} → {pred_str} {status}")

    total += 1
    if pred_str == label:
        correct += 1

# === Итоги
acc = (correct / total) * 100 if total else 0
print(f"\n🎯 Accuracy: {acc:.2f}% ({correct}/{total})")
