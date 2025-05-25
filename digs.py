from PIL import Image
import os

# === ПАРАМЕТРЫ СЕТКИ ===
rows = 3
cols = 7

# === СПИСОК СИМВОЛОВ (по порядку в таблице) ===
symbols = list("0123456789!?,;-+():\"%")

# === КАРТА БЕЗОПАСНЫХ ИМЁН ===
def sanitize(symbol):
    replacements = {
        " ": "space",
        ":": "colon",
        "\"": "quote",
        "%": "percent",
        "?": "question",
        "!": "exclam",
        ",": "comma",
        ".": "dot",
        ";": "semicolon",
        "+": "plus",
        "-": "minus",
        "(": "lparen",
        ")": "rparen"
    }
    return replacements.get(symbol, symbol)

# === СОЗДАНИЕ ПАПКИ ===
os.makedirs("dataset/symbols", exist_ok=True)

# === ОБРАБОТКА ИЗОБРАЖЕНИЯ ===
img = Image.open("images/symbols.jpg").convert("L")
w, h = img.size
cell_w = w // cols
cell_h = h // rows

idx = 0
for row in range(rows):
    for col in range(cols):
        if idx >= len(symbols):
            break
        x0 = col * cell_w
        y0 = row * cell_h
        x1 = x0 + cell_w
        y1 = y0 + cell_h
        roi = img.crop((x0, y0, x1, y1))

        symbol = symbols[idx]
        filename = sanitize(symbol) + ".jpg"

        roi.save(os.path.join("dataset/symbols", filename))
        idx += 1

print("✅ Готово! Знаки сохранены в dataset/symbols/")
