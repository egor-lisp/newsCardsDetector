import cv2
from boxes_exctractor import extract_boxes
from cards_finder import PatternMatcher
import time


def draw_boxes(img, boxes, output_path, color=(0, 0, 255)):
    """Рисует простые боксы (x, y, w, h)."""
    boxed_img = img.copy()
    for idx, (x, y, w, h) in enumerate(boxes, 1):
        cv2.rectangle(boxed_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(boxed_img, str(idx), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(output_path, boxed_img)


def draw_matches(img, matches, output_path):
    """Рисует найденные карточки (каждая = [media, image, description])."""
    boxed_img = img.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # media, image, description

    for idx, match in enumerate(matches, 1):
        for box, color in zip(match, colors):
            x, y, w, h = box
            cv2.rectangle(boxed_img, (x, y), (x + w, y + h), color, 2)

        # рамка вокруг всей карточки
        x_min = min(b[0] for b in match)
        y_min = min(b[1] for b in match)
        x_max = max(b[0] + b[2] for b in match)
        y_max = max(b[1] + b[3] for b in match)
        cv2.rectangle(boxed_img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        cv2.putText(boxed_img, str(idx), (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imwrite(output_path, boxed_img)


def main():
    pattern_matcher = PatternMatcher()
    s = time.time()
    filtered_boxes, _, img = extract_boxes("test_images/11.png")
    print(f"Время выполнения extract_boxes: {time.time() - s}")

    # сохраняем картинку только с filtered_boxes
    draw_boxes(img, filtered_boxes, "result.png")

    matches_v = pattern_matcher.pattern_news_card_vertical(filtered_boxes)
    print("Найдено карточек по вертикальному паттерну:", len(matches_v))
    draw_matches(img, matches_v, "boxed_cards.png")

    matches_h = pattern_matcher.pattern_news_card_side(filtered_boxes)
    print("Найдено карточек по горизонтальному паттерну:", len(matches_h))
    draw_matches(img, matches_h, "boxed_cards_side.png")


if __name__ == "__main__":
    main()
