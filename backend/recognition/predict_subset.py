import os
import sys
from PIL import Image
from recognition.ocr_model import CrnnOcrModel

# === Пути
weights_path = "backend/recognition/weights/crnn_weights.pth"
alphabet_path = "backend/recognition/weights/alphabet.txt"

# === Модель
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

def evaluate_folders(dataset_root, folders):
    print(f"\n📂 Проверка указанных папок: {', '.join(folders)}")
    total = 0
    correct = 0
    for label in folders:
        folder_path = os.path.join(dataset_root, label)
        if not os.path.isdir(folder_path):
            print(f"⚠️ Папка не найдена: {folder_path}")
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

def evaluate_tsv(tsv_path, image_folder):
    print(f"\n📄 Проверка TSV: {tsv_path}")
    if not os.path.exists(tsv_path):
        print(f"❌ Не найден: {tsv_path}")
        return
    total = 0
    correct = 0
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            if '\t' not in line:
                continue
            parts = line.strip().split("\t")
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
        print("⚠️ Нет подходящих записей.")
    else:
        print(f"🎯 Точность: {correct}/{total} = {correct/total*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("🔧 Использование:")
        print("  python predict_subset.py А Ё I        — проверить только указанные папки")
        print("  python predict_subset.py --tsv        — проверить только train.tsv и test.tsv")
        print("  python predict_subset.py А Б --tsv    — и буквы, и TSV")
        sys.exit(0)

    root = "dataset"
    folders = []
    check_tsv = False

    for arg in sys.argv[1:]:
        if arg == "--tsv":
            check_tsv = True
        else:
            folders.append(arg)

    if folders:
        evaluate_folders(root, folders)

    if check_tsv:
        evaluate_tsv(os.path.join(root, "train.tsv"), os.path.join(root, "train"))
        evaluate_tsv(os.path.join(root, "test.tsv"), os.path.join(root, "test"))
