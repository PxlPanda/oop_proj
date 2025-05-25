import os
import cv2

# === Настройки ===
input_images = {
    "images/alphabet.jpg": list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"),
    "images/digits_and_text.jpg": list("0123456789!?,;-+():«»%")
}

grids = {
    "images/alp.jpg": (10, 7),
    "images/dig": (3, 7)
}

output_root = "dataset"

# === Функция ===
def split_grid_and_save(image_path, labels, grid_rows, grid_cols, output_root):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Не удалось открыть: {image_path}")
        return

    h, w = img.shape[:2]
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    assert len(labels) <= grid_rows * grid_cols, "⚠️ Слишком мало ячеек!"

    i = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if i >= len(labels): break
            x, y = col * cell_w, row * cell_h
            roi = img[y:y+cell_h, x:x+cell_w]
            label = labels[i]
            folder = os.path.join(output_root, label)
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, f"{label}_{i}.png")
            cv2.imwrite(filename, roi)
            i += 1
    print(f"✅ {image_path} → {i} символов сохранено")

# === Запуск ===
for path, lbls in input_images.items():
    split_grid_and_save(path, lbls, *grids[path], output_root)
