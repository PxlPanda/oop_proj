import cv2
import numpy as np

def split_into_chars(image_path, grid_size=(6, 6)):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cols, rows = grid_size
    cell_w = w // cols
    cell_h = h // rows

    chars = []
    for row in range(rows):
        for col in range(cols):
            x = col * cell_w
            y = row * cell_h
            roi = bw[y:y + cell_h, x:x + cell_w]
            chars.append((x, y, cell_w, cell_h, roi))
    return chars
