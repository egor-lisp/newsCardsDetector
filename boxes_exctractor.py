import cv2
import numpy as np

def extract_boxes(image_or_path):
    # ---------- 1. Загрузка ----------
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path)
    else:
        img = image_or_path.copy()
    height, width, _ = img.shape

    # ---------- 2. Фон через подвыборку ----------
    pixels = img.reshape(-1, 3)
    if len(pixels) > 200_000:
        idx = np.random.choice(len(pixels), 200_000, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    # безопасный быстрый способ поиска наиболее частого цвета
    # объединяем RGB в одно число: R*256^2 + G*256 + B
    sample_code = sample[:,0].astype(np.int32) * 256*256 + \
                  sample[:,1].astype(np.int32) * 256 + \
                  sample[:,2].astype(np.int32)
    bg_code = np.bincount(sample_code).argmax()
    # восстанавливаем цвет
    r = (bg_code >> 16) & 255
    g = (bg_code >> 8) & 255
    b = bg_code & 255
    bg_color = np.array([r, g, b], dtype=np.uint8)
    bg_brightness = bg_color.mean()

    # ---------- 3. Маски ----------
    pix_brightness = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask1 = (bg_brightness < 128) & (pix_brightness > bg_brightness + 50)
    mask2 = (bg_brightness >= 128) & (pix_brightness < bg_brightness - 50)
    mask3 = np.abs(pix_brightness - bg_brightness) > 30

    result = img.copy()
    result[mask1 | mask2] = [255, 0, 0]
    result[mask3] = [0, 0, 255]

    blue_mask = (mask1 | mask2).astype(np.uint8) * 255

    # ---------- 4. Морфология и контуры ----------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blue_mask_dilated = cv2.dilate(blue_mask, kernel, iterations=3)
    contours, _ = cv2.findContours(blue_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]

    # ---------- 5. Фильтрация вложенных боксов ----------
    boxes_np = np.array(boxes)
    filtered_boxes = []
    for i, (x1, y1, w1, h1) in enumerate(boxes_np):
        inside = np.any(
            (boxes_np[:,0] <= x1) &
            (boxes_np[:,1] <= y1) &
            ((boxes_np[:,0]+boxes_np[:,2]) >= x1+w1) &
            ((boxes_np[:,1]+boxes_np[:,3]) >= y1+h1) &
            (np.arange(len(boxes_np)) != i)
        )
        if not inside and w1 > 10 and h1 > 10:
            filtered_boxes.append((x1, y1, w1, h1))

    # ---------- 6. Отрисовка боксов ----------
    boxed = img.copy()
    for (x, y, w, h) in filtered_boxes:
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return filtered_boxes, result, img
