from PIL import Image
import os

# === ПАРАМЕТРЫ СЕТКИ ===
rows = 5
cols = 7

# === СПИСКИ БУКВ ===
alphabet_upper = [
    "А", "Б", "В", "Г", "Д", "Е", "Ё",
    "Ж", "З", "И", "Й", "К", "Л", "М",
    "Н", "О", "П", "Р", "С", "Т", "У",
    "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ",
    "Ы", "Ь", "Э", "Ю", "Я"
]

alphabet_lower = [
    "а", "б", "в", "г", "д", "е", "ё",
    "ж", "з", "и", "й", "к", "л", "м",
    "н", "о", "п", "р", "с", "т", "у",
    "ф", "х", "ц", "ч", "ш", "щ", "ъ",
    "ы", "ь", "э", "ю", "я"
]

# === СОЗДАНИЕ ПАПОК ===
os.makedirs("dataset/uppercase", exist_ok=True)
os.makedirs("dataset/lowercase", exist_ok=True)

def process_image(image_path, alphabet, output_dir):
    img = Image.open(image_path).convert("L")
    w, h = img.size
    cell_w = w // cols
    cell_h = h // rows

    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= len(alphabet):
                break
            x0 = col * cell_w
            y0 = row * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            roi = img.crop((x0, y0, x1, y1))
            symbol = alphabet[idx]
            safe_name = symbol if symbol != " " else "space"
            roi.save(os.path.join(output_dir, f"{safe_name}.jpg"))
            idx += 1

# === ЗАПУСК ===
process_image("images/uppercase.jpg", alphabet_upper, "dataset/uppercase")
process_image("images/lowercase.jpg", alphabet_lower, "dataset/lowercase")

print("✅ Готово! Файлы сохранены в папку dataset/")
