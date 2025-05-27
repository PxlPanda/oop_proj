import sys
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from recognition.crnn_model import CRNN

# === Аргументы ===
if len(sys.argv) != 2:
    print("Укажи путь к изображению, например: python3 predict_crnn.py dataset/words/привет.jpg")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]
WEIGHTS_PATH = "backend/recognition/weights/crnn_weights.pth"
ALPHABET_PATH = "backend/recognition/weights/alphabet.txt"

# === Загрузка алфавита из файла ===
if not os.path.exists(ALPHABET_PATH):
    print(f"Файл алфавита не найден: {ALPHABET_PATH}")
    sys.exit(1)

with open(ALPHABET_PATH, encoding="utf-8") as f:
    alphabet = list(f.read())

idx_to_char = {i: c for i, c in enumerate(alphabet)}
nclass = len(alphabet) + 1  # +1 для blank

print("Алфавит:", ''.join(alphabet))
print("Классы (включая blank):", nclass)

# === Подготовка изображения ===
transform = transforms.Compose([
    transforms.Resize((32, 100)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = transform(img).unsqueeze(0)  # [1, 1, 32, 100]
print("🖼 Размер тензора:", img_tensor.shape)

# === Загрузка модели ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CRNN(32, 1, nclass, 256).to(device)

checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# === Предсказание ===
with torch.no_grad():
    img_tensor = img_tensor.to(device)
    output = model(img_tensor)  # [T, B, C]
    log_probs = F.log_softmax(output, dim=2)
    pred_indices = log_probs.argmax(2).squeeze(1).cpu().numpy()

print("Индексы предсказания:", pred_indices)

# === CTC-декодирование (удаляем повторы и blank)
result = []
prev = -1
for idx in pred_indices:
    if idx != prev and idx != nclass - 1:
        result.append(idx_to_char.get(idx, "?"))
    prev = idx

print("Распознано:", ''.join(result))
