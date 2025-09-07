import cv2
import numpy as np
from collections import Counter

# Загружаем скрин
img = cv2.imread("11.png")

# Находим самый частый цвет (фон)
pixels = img.reshape(-1, 3)
pixels_tuple = [tuple(p) for p in pixels]
most_common_color = Counter(pixels_tuple).most_common(1)[0][0]
bg_color = np.array(most_common_color, dtype=np.uint8)

# Функция яркости
def brightness(c):
    return np.mean(c)

bg_brightness = brightness(bg_color)

# Создаём результат и маску синих пикселей
result = np.zeros_like(img)
height, width, _ = img.shape
blue_mask = np.zeros((height, width), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        pix = img[y, x]
        pix_brightness = brightness(pix)
        if bg_brightness < 128 and pix_brightness > bg_brightness + 50:
            result[y, x] = [255, 0, 0]
            blue_mask[y, x] = 255
        elif bg_brightness >= 128 and pix_brightness < bg_brightness - 50:
            result[y, x] = [255, 0, 0]
            blue_mask[y, x] = 255
        elif abs(pix_brightness - bg_brightness) > 30:
            result[y, x] = [0, 0, 255]
        else:
            result[y, x] = pix

# Морфологическая обработка для объединения близких синих пикселей
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
blue_mask_dilated = cv2.dilate(blue_mask, kernel, iterations=3)

# Находим контуры объединённых областей
contours, _ = cv2.findContours(blue_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Собираем все bounding boxes
boxes = [cv2.boundingRect(cnt) for cnt in contours]

# Фильтруем вложенные боксы
filtered_boxes = []
for i, (x1, y1, w1, h1) in enumerate(boxes):
    inside = False
    for j, (x2, y2, w2, h2) in enumerate(boxes):
        if i == j:
            continue
        if x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2):
            inside = True
            break
    if not inside:
        filtered_boxes.append((x1, y1, w1, h1))

# Обводим отфильтрованные bounding boxes на исходном изображении
boxed = img.copy()
for (x, y, w, h) in filtered_boxes:
    if w > 10 and h > 10:  # фильтр мелких шумов
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Сохраняем результаты
cv2.imwrite("highlighted.png", result)
cv2.imwrite("boxed.png", boxed)
