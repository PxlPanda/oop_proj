import os
from PIL import Image
from recognition.ocr_model import CrnnOcrModel
import torch

weights_path = "backend/recognition/weights/crnn_weights.pth"
alphabet_path = "backend/recognition/weights/alphabet.txt"

ocr = CrnnOcrModel(weights_path=weights_path, alphabet_path=alphabet_path)

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

def evaluate_tsv(tsv_path, image_folder):
    print(f"\n Проверка: {tsv_path}")
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
    print(f"🎯 Точность: {correct}/{total} = {correct/total*100:.2f}%" if total else "⚠️ Нет данных.")

def evaluate_char_dirs(root_dir):
    print(f"\n📁 Проверка папок с одиночными символами: {root_dir}")
    total = 0
    correct = 0
    for label in os.listdir(root_dir):
        folder = os.path.join(root_dir, label)
        if not os.path.isdir(folder) or len(label) != 1:
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            path = os.path.join(folder, fname)
            if predict_and_check(path, label):
                correct += 1
            total += 1
    print(f"🎯 Точность: {correct}/{total} = {correct/total*100:.2f}%" if total else "⚠️ Нет изображений.")

if __name__ == "__main__":
    evaluate_char_dirs("dataset")
    evaluate_tsv("dataset/train.tsv", "dataset/train")
    evaluate_tsv("dataset/test.tsv", "dataset/test")
