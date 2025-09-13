import cv2
import numpy as np
import os
import json
import easyocr
from boxes_exctractor import extract_boxes

OUTPUT_DIR = "cards"
os.makedirs(OUTPUT_DIR, exist_ok=True)

JSON_PATH = os.path.join(OUTPUT_DIR, "cards.json")

reader = easyocr.Reader(['ru', 'en'], gpu=True)  # если есть CUDA — gpu=True

# параметры фильтрации
PERSIST_FRAMES = 3
SSIM_MATCH_PENDING = 0.85
SSIM_MATCH_SAVED = 0.85
MIN_CARD_HEIGHT = 40
PRUNE_AGE = 60


def find_vertical_ellipsis(boxes):
    """Ищем боксы кнопок-меню (троеточий)."""
    ellipses = []
    for (x, y, w, h) in boxes:
        if w > 0 and h >= 1.5 * w and w < 40 and h < 120:
            ellipses.append((x, y, w, h))
    return ellipses


def prepare_repr(img):
    """Грейскейл + ресайз для сравнения карточек."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        r = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA)
    except Exception:
        return None
    return r.astype(np.float32)


def ssim(img1, img2):
    """Вычисление SSIM."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq, mu2_sq = mu1 * mu1, mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    return float(np.mean(ssim_map))


def ocr_card(img, fname):
    """OCR карточки: сверху — название медиа, снизу — текст статьи."""
    H, W = img.shape[:2]
    media_region = img[0:int(0.25 * H), 0:int(0.6 * W)]  # верхняя левая часть
    article_region = img[int(0.6 * H):H, 0:W]            # нижняя часть

    media_texts = reader.readtext(media_region, detail=0, paragraph=True)
    article_texts = reader.readtext(article_region, detail=0, paragraph=True)

    media_text = " ".join(media_texts).strip()
    article_text = " ".join(article_texts).strip()

    return {
        "file": fname,
        "media": media_text,
        "article": article_text
    }


def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео.")
        return

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    saved_count = 0

    pending = []
    saved_reprs = []
    json_records = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        filtered_boxes, _, img = extract_boxes(frame)
        ellipses = find_vertical_ellipsis(filtered_boxes)
        ellipses_sorted = sorted(ellipses, key=lambda b: b[1])

        candidates = []
        # ✅ карточки строго между соседними кнопками-меню
        if len(ellipses_sorted) >= 2:
            for i in range(len(ellipses_sorted) - 1):
                top = ellipses_sorted[i]
                bot = ellipses_sorted[i + 1]

                y_start = top[1] + top[3]  # низ верхнего троеточия
                y_end = bot[1]             # верх нижнего троеточия

                if y_end - y_start < MIN_CARD_HEIGHT:
                    continue

                card = img[y_start:y_end, :]
                if card.size == 0:
                    continue

                candidates.append({'y_range': (y_start, y_end), 'card': card})

        seen_this_frame = set()
        for cand in candidates:
            cand_repr = prepare_repr(cand['card'])
            if cand_repr is None:
                continue

            matched = False
            for p in pending:
                score = ssim(p['repr'], cand_repr)
                if score >= SSIM_MATCH_PENDING:
                    p['last_img'] = cand['card']
                    p['repr'] = (p['repr'] * p['count'] + cand_repr) / (p['count'] + 1)
                    p['count'] += 1
                    p['last_seen_frame'] = frame_count
                    p['y_range'] = cand['y_range']
                    matched = True
                    seen_this_frame.add(id(p))
                    break

            if not matched:
                new_p = {
                    'repr': cand_repr,
                    'last_img': cand['card'],
                    'count': 1,
                    'last_seen_frame': frame_count,
                    'y_range': cand['y_range']
                }
                pending.append(new_p)
                seen_this_frame.add(id(new_p))

        # чистим устаревшие pending
        pending = [p for p in pending if frame_count - p['last_seen_frame'] <= PRUNE_AGE]

        # сохраняем устоявшиеся карточки
        to_remove_ids = []
        for p in pending:
            if p['count'] >= PERSIST_FRAMES:
                is_dup = False
                for srepr in saved_reprs:
                    if ssim(srepr, p['repr']) >= SSIM_MATCH_SAVED:
                        is_dup = True
                        break
                if not is_dup:
                    saved_count += 1
                    fname = os.path.join(OUTPUT_DIR, f"card_{saved_count}.png")
                    cv2.imwrite(fname, p['last_img'])
                    saved_reprs.append(p['repr'])

                    record = ocr_card(p['last_img'], fname)
                    json_records.append(record)

                    print(f"[frame {frame_count}] Сохранена карточка: {fname}")
                to_remove_ids.append(id(p))

        pending = [p for p in pending if id(p) not in to_remove_ids]

        if frame_count % 100 == 0 or frame_count == total_frames:
            print(f"Кадр {frame_count}/{total_frames}  pending={len(pending)}  saved={saved_count}")

    cap.release()

    # итоговый JSON
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_records, f, ensure_ascii=False, indent=2)

    print("Готово. Сохранено карточек:", saved_count)
    print("JSON:", JSON_PATH)


if __name__ == "__main__":
    process_video("video.mp4")
