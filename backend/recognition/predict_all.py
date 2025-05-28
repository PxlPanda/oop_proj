import os
from PIL import Image
from recognition.ocr_model import CrnnOcrModel
import torch

with open("backend/recognition/weights/alphabet.txt", encoding="utf-8") as f:
    print("✔️ len(alphabet):", len(f.read()))

# === Настройки
weights_path = "backend/recognition/weights/crnn_weights.pth"
alphabet_path = "backend/recognition/weights/alphabet.txt"

ocr = CrnnOcrModel(weights_path=weights_path, alphabet_path=alphabet_path)

checkpoint = torch.load(weights_path, map_location="cpu")
model_dict = checkpoint["model_state_dict"]
nclass_from_weights = model_dict["rnn.1.embedding.weight"].shape[0]
print(f"🧠 Модель в весах обучена на nclass = {nclass_from_weights}")

def predict_and_check(img_path, expected_label):
    try:
        img = Image.open(img_path).convert("RGB")
        pred = ocr.predict(img)
        result = "✅" if pred == expected_label else "❌"
        print(f"{os.path.basename(img_path)} → {pred} ({expected_label}) {result}")
        return pred == expected_label
    except Exception as e:
        print(f"{img_path} — ❌ Ошибка: {e}")
        return False

def evaluate_folder(folder):
    print(f"\nПроверка папки: {folder}")
    total = 0
    correct = 0
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        label = os.path.splitext(fname)[0]
        path = os.path.join(folder, fname)
        if predict_and_check(path, label):
            correct += 1
        total += 1
    if total == 0:
        print("Нет подходящих файлов.")
    else:
        print(f"🎯 Точность: {correct}/{total} = {correct/total*100:.2f}%")

def evaluate_tsv(tsv_path, image_folder):
    print(f"\n📄 Проверка по TSV: {tsv_path}")
    if not os.path.exists(tsv_path):
        print(f"⚠️ TSV-файл не найден: {tsv_path}")
        return
    total = 0
    correct = 0
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            if '\t' not in line:
                continue
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            fname, label = parts
            img_path = os.path.join(image_folder, fname)
            if not os.path.exists(img_path):
                continue
            if predict_and_check(img_path, label):
                correct += 1
            total += 1
    if total == 0:
        print("Нет валидных пар.")
    else:
        print(f"Точность: {correct}/{total} = {correct/total*100:.2f}%")

def evaluate_char_folders(root_dir):
    print(f"\n Проверка по папкам-символам: {root_dir}")
    total = 0
    correct = 0
    for label in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, label)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(folder_path, fname)
            if predict_and_check(img_path, label):
                correct += 1
            total += 1
    if total == 0:
        print("⚠️ Нет подходящих изображений.")
    else:
        print(f"🎯 Точность: {correct}/{total} = {correct/total*100:.2f}%")


if __name__ == "__main__":
    evaluate_char_folders("dataset")
    evaluate_tsv("dataset/train.tsv", "dataset/train")
    evaluate_tsv("dataset/test.tsv", "dataset/test")
