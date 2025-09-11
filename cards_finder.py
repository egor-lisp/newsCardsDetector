
class PatternMatcher:
    def __init__(self, col_tol=20):
        self.col_tol = col_tol  # допустимое отклонение по X для "одной колонки"
        
        self.y_align_tol = 20
        self.max_gap_source_desc = 160
        self.min_desc_h_px = 40
        self.desc_min_area_vs_source = 3.0
        self.desc_min_w_vs_source = 1.2
        self.desc_min_h_vs_source = 1.5
        self.desc_min_w_vs_img = 0.35

    # === ПАТТЕРН 1: вертикальный (источник → картинка → описание) ===
    def pattern_news_card_vertical(self, rects):
        matches = []
        rects_sorted = sorted(rects, key=lambda r: (r[1], r[0]))  # сверху вниз

        for i in range(len(rects_sorted) - 2):
            source = rects_sorted[i]
            img = rects_sorted[i + 1]
            desc = rects_sorted[i + 2]

            x1, y1, w1, h1 = source
            x2, y2, w2, h2 = img
            x3, y3, w3, h3 = desc

            if not (w1 > 2 * h1 and h1 < 50):  # media box
                continue
            if not (w2 > w1 * 0.7 and h2 > h1 * 2):  # image box
                continue
            if not (y3 > y2 + h2):  # description ниже картинки
                continue
            if not (0.6 * w2 < w3 < 1.3 * w2):  # ширина описания примерно как у картинки
                continue

            def same_column(r1, r2, tol=self.col_tol):
                return abs(r1[0] - r2[0]) < tol

            # if not (same_column(source, img) and same_column(img, desc)):
            #     continue

            matches.append([source, img, desc])

        return matches

    # === ПАТТЕРН 2: слева источник, под ним большое описание; справа картинка ===
    def pattern_news_card_side(self, rects):
        matches = []
        rects_sorted = sorted(rects, key=lambda r: (r[1], r[0]))

        def is_source(r):
            x, y, w, h = r
            return (w > 2 * h) and (h < 50)

        def same_column(r1, r2, tol=self.col_tol):
            return abs(r1[0] - r2[0]) < tol

        def area(r):
            return r[2] * r[3]

        used_triplets = set()

        for i, source in enumerate(rects_sorted):
            if not is_source(source):
                continue

            x1, y1, w1, h1 = source
            src_bottom = y1 + h1

            image_candidates = []
            for k, img in enumerate(rects_sorted):
                if k == i:
                    continue
                x3, y3, w3, h3 = img
                if x3 <= x1 + w1:
                    continue
                if y3 > y1 + h1 + self.y_align_tol:
                    continue
                image_candidates.append((k, img))

            if not image_candidates:
                continue

            for k, img in image_candidates:
                x3, y3, w3, h3 = img
                desc_candidates = []
                for j, desc in enumerate(rects_sorted):
                    if j in (i, k):
                        continue
                    x2, y2, w2, h2 = desc
                    if not (y2 > src_bottom and same_column(source, desc)):
                        continue
                    if (y2 - src_bottom) > self.max_gap_source_desc:
                        continue
                    if h2 < max(self.min_desc_h_px, self.desc_min_h_vs_source * h1):
                        continue
                    if w2 < self.desc_min_w_vs_source * w1:
                        continue
                    if area(desc) < self.desc_min_area_vs_source * area(source):
                        continue
                    if w2 < self.desc_min_w_vs_img * w3:
                        continue
                    if not (x3 >= x2 + w2 + 5):
                        continue
                    if (y3 + h3) < y2:
                        continue

                    desc_candidates.append((j, desc))

                if not desc_candidates:
                    continue

                j_best, desc_best = max(desc_candidates, key=lambda t: (area(t[1]), t[1][3]))

                triplet = (source, desc_best, img)
                key = (tuple(source), tuple(desc_best), tuple(img))
                if key not in used_triplets:
                    used_triplets.add(key)
                    matches.append([source, desc_best, img])

        return matches
