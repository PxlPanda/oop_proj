# inspect_weights.py
import os
import torch
import reprlib

alphabet_path = "backend/recognition/weights/alphabet.txt"
weights_path = "backend/recognition/weights/crnn_weights.pth"

# === Проверка наличия alphabet.txt
if not os.path.exists(alphabet_path):
    print(f"❌ Не найден файл алфавита: {alphabet_path}")
    exit(1)

# === Чтение alphabet.txt (без лишних пробелов и переносов)
with open(alphabet_path, encoding="utf-8") as f:
    alphabet_raw = f.read()

print(f"📖 Содержимое alphabet.txt (сокращённо): {reprlib.repr(alphabet_raw)}")

alphabet = alphabet_raw.strip("\r\n ")  # удалить пробелы и переносы по краям

print(f"📖 Количество символов в alphabet.txt (после очистки): {len(alphabet)}")

# === Ожидаемое число классов модели (с учётом blank для CTC)
nclass_expected = len(alphabet) + 1
print(f"🔠 Ожидаемое nclass для модели (alphabet + blank): {nclass_expected}")

# === Проверка наличия файла весов
if not os.path.exists(weights_path):
    print(f"❌ Не найден файл весов модели: {weights_path}")
    exit(1)

# === Загрузка весов модели
checkpoint = torch.load(weights_path, map_location="cpu")

if "model_state_dict" not in checkpoint:
    print("❌ В файле весов нет ключа 'model_state_dict'")
    exit(1)

state_dict = checkpoint["model_state_dict"]

# === Проверяем ключ с embedding весами, ищем тот, что содержит nclass
embedding_keys = [k for k in state_dict.keys() if "embedding.weight" in k]

if not embedding_keys:
    print("❌ Не найден ключ с embedding весами в модели")
    exit(1)

# Возьмём первый подходящий ключ (обычно rnn.1.embedding.weight)
embedding_key = embedding_keys[0]
nclass_actual = state_dict[embedding_key].shape[0]

print(f"📦 nclass из весов модели (по ключу '{embedding_key}'): {nclass_actual}")

# === Сравнение ожидаемого и фактического nclass
if nclass_expected == nclass_actual:
    print("✅ Всё совпадает: размеры алфавита и весов модели согласованы.")
else:
    print("❌ Несовпадение размеров!")
    print(f"   alphabet.txt содержит {len(alphabet)} символов (+1 blank = {nclass_expected}),")
    print(f"   но в весах модели nclass = {nclass_actual}.")
    print("   Возможно, alphabet.txt был изменён после обучения или веса не соответствуют алфавиту.")
