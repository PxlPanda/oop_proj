from .segment import split_into_chars
from .ocr_model import CrnnOcrModel
from api.mongo_models import Symbol
from django.conf import settings
import os

def recognize_template(session):
    image_path = os.path.join(settings.MEDIA_ROOT, session.image_path)
    print("📂 Путь к изображению:", image_path)
    print("📂 Файл существует?", os.path.exists(image_path))

    if not os.path.exists(image_path):
        print("❌ Файл не найден. Прерываю.")
        return

    # Сегментация
    char_cells = split_into_chars(image_path)
    print(f"🧩 Сегментировано символов: {len(char_cells)}")

    if not char_cells:
        print("⚠️ Ни один символ не найден. Проверь split_into_chars().")
        return

    # OCR-модель
    ocr = CrnnOcrModel(weights_path='recognition/weights/crnn_weights.pth')
    print("📦 Модель загружена. Начинаю распознавание...")

    for x, y, w, h, roi in char_cells:
        try:
            char = ocr.predict(roi)
            print(f"→ ROI ({x},{y}) → '{char}'")
            s = Symbol.objects.create(
                session=session,
                x=x, y=y,
                width=w, height=h,
                label=char,
                is_corrected=False
            )
            print("✅ Символ сохранён:", s.id)
        except Exception as e:
            print("❌ Ошибка при распознавании или сохранении:", e)

    session.is_verified = False
    session.save()
    print("🎉 Сессия завершена и сохранена.")
