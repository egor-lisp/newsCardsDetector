import json
import os


def count_real_lines(text):
    """
    Подсчитывает реальное количество строк в тексте.
    Использует несколько методов разделения для более точного подсчета.
    """
    if not text:
        return 0
    
    # Метод 1: по явным переносам строк
    lines1 = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Метод 2: по точкам и другим знакам препинания
    lines2 = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    
    # Метод 3: по большим пробелам (возможные визуальные разделители)
    lines3 = [line.strip() for line in text.split('  ') if line.strip()]
    
    # Берем максимальное количество строк из всех методов
    max_lines = max(
        len([l for l in lines1 if any(c.isalnum() for c in l)]),
        len([l for l in lines2 if any(c.isalnum() for c in l)]),
        len([l for l in lines3 if any(c.isalnum() for c in l)])
    )
    
    return max_lines


def validate_cards_json(json_path, min_lines=2):
    """
    Проверяет JSON-файл с карточками и создает новый, только с валидными карточками.
    """
    # Читаем оригинальный JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    
    valid_cards = []
    invalid_files = []
    
    for card in cards:
        article_text = card.get('article', '')
        line_count = count_real_lines(article_text)
        
        if line_count >= min_lines:
            valid_cards.append(card)
        else:
            invalid_files.append(card['file'])
    
    # Создаем новый JSON только с валидными карточками
    output_json = json_path.replace('.json', '_validated.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(valid_cards, f, ensure_ascii=False, indent=2)
    
    # Удаляем файлы изображений невалидных карточек
    for file_path in invalid_files:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    print(f"Всего карточек: {len(cards)}")
    print(f"Валидных карточек: {len(valid_cards)}")
    print(f"Отфильтровано карточек: {len(cards) - len(valid_cards)}")
    print(f"Результат сохранен в: {output_json}")
    
    return valid_cards


if __name__ == "__main__":
    # Путь к JSON-файлу с карточками
    json_path = os.path.join("cards", "cards.json")
    validate_cards_json(json_path, min_lines=2)
