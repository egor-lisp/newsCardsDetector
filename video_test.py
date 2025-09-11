import cv2
import numpy as np
import os
from boxes_exctractor import extract_boxes

OUTPUT_DIR = "cards"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Параметры — подстраивай
PERSIST_FRAMES = 3         # сколько кадров подряд кандидат должен существовать
SSIM_MATCH_PENDING = 0.88  # если SSIM >= этого — кандидат считается тем же pending
SSIM_MATCH_SAVED = 0.86    # если SSIM >= этого — считается дубликатом уже сохранённого
MIN_CARD_HEIGHT = 80       # минимальная высота вырезаемой карточки (px)
PRUNE_AGE = 40             # удалять pending, если не виделись столько кадров

def find_vertical_ellipsis(boxes):
    ellipses = []
    for (x, y, w, h) in boxes:
        if w > 0 and h >= 1.5 * w and w < 40 and h < 120:  # можно подстроить
            ellipses.append((x, y, w, h))
    return ellipses

def prepare_repr(img):
    """Грейскейл + ресайз в фиксированный размер для стабильного сравнения."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        r = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA)
    except Exception:
        return None
    return r.astype(np.float32)

def ssim(img1, img2):
    """SSIM для двух grayscale float32 изображений (0..255)."""
    # ожидается одинаковый размер
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    # предотвратить деление на ноль
    ssim_map = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    return float(np.mean(ssim_map))

def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео.")
        return

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    saved_count = 0

    pending = []  # список dict: {'repr', 'last_img', 'count', 'last_seen_frame', 'y_range'}
    saved_reprs = []  # список repr для сравнения (grayscale 256x256)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        filtered_boxes, _, img = extract_boxes(frame)
        ellipses = find_vertical_ellipsis(filtered_boxes)
        ellipses_sorted = sorted(ellipses, key=lambda b: b[1])

        # формируем кандидаты — между соседними троеточиями в текущем кадре
        candidates = []
        for i in range(len(ellipses_sorted) - 1):
            top = ellipses_sorted[i]
            bot = ellipses_sorted[i + 1]
            y_start = top[1] + top[3]
            y_end = bot[1]
            if y_end - y_start < MIN_CARD_HEIGHT:
                continue
            card = img[y_start:y_end, :]
            if card.size == 0:
                continue
            candidates.append({'y_range': (y_start, y_end), 'card': card})

        # Обработка кандидатов: сопоставляем с pending по SSIM
        seen_this_frame = set()
        for cand in candidates:
            cand_repr = prepare_repr(cand['card'])
            if cand_repr is None:
                continue

            matched = False
            # ищем подходящий pending
            for p in pending:
                score = ssim(p['repr'], cand_repr)
                if score >= SSIM_MATCH_PENDING:
                    # обновляем pending
                    p['last_img'] = cand['card']
                    p['repr'] = (p['repr'] * p['count'] + cand_repr) / (p['count'] + 1)  # усреднение
                    p['count'] += 1
                    p['last_seen_frame'] = frame_count
                    p['y_range'] = cand['y_range']
                    matched = True
                    seen_this_frame.add(id(p))
                    break

            if not matched:
                # создаём новый pending
                new_p = {
                    'repr': cand_repr,
                    'last_img': cand['card'],
                    'count': 1,
                    'last_seen_frame': frame_count,
                    'y_range': cand['y_range']
                }
                pending.append(new_p)
                seen_this_frame.add(id(new_p))

        # Удаляем pending, которые давно не виделись
        pending = [p for p in pending if frame_count - p['last_seen_frame'] <= PRUNE_AGE]

        # Проверяем pending, которые достигли устойчивости — сохраняем если уникальны
        to_remove_ids = []
        for p in pending:
            if p['count'] >= PERSIST_FRAMES:
                # сравнить с уже сохранёнными
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
                    print(f"[frame {frame_count}] Сохранена карточка: {fname}")
                # в любом случае помечаем для удаления
                to_remove_ids.append(id(p))

        # удаляем все помеченные по id
        pending = [p for p in pending if id(p) not in to_remove_ids]

        if frame_count % 100 == 0 or frame_count == total_frames:
            print(f"Кадр {frame_count}/{total_frames}  pending={len(pending)}  saved={saved_count}")

    cap.release()
    print("Готово. Сохранено карточек:", saved_count)


if __name__ == "__main__":
    process_video("video.mp4")
