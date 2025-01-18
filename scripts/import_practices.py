import re
import httpx
import asyncio
from pathlib import Path
import logging
from datetime import datetime

# Настраиваем логирование в файл и консоль
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"import_practices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Файлы для отслеживания прогресса
success_file = log_dir / "successful_practices.txt"
failed_practices_file = log_dir / "failed_practices.txt"

def normalize_title(title: str) -> str:
    """Нормализует название практики для сравнения."""
    # Убираем артикли и лишние пробелы
    title = re.sub(r'^(The|A|An)\s+', '', title.strip())
    # Убираем апострофы и специальные символы
    title = re.sub(r'[\'"`]', '', title)
    # Приводим к нижнему регистру
    return title.lower()

# Загружаем список успешно созданных практик
successful_practices = set()
if success_file.exists():
    successful_practices = set(normalize_title(title) for title in success_file.read_text(encoding='utf-8').splitlines())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def parse_practice(practice_text: str) -> dict:
    """Извлекает данные практики из текста."""
    # Извлекаем заголовок
    title_match = re.search(r'### \d+\. (.+)', practice_text)
    if not title_match:
        return None
    
    title = title_match.group(1)
    
    # Извлекаем остальные поля
    fields = {}
    field_pattern = r'\*\*(\w+)\*\*: "?([^"\n]+)"?'
    for match in re.finditer(field_pattern, practice_text):
        field_name, value = match.groups()
        fields[field_name.lower()] = value.strip()
    
    # Формируем данные для API
    practice_data = {
        "title": title,
        "description": fields.get('problem', ''),
        "domain": fields.get('domain', ''),
        "tags": [tag.strip() for tag in fields.get('sub_domains', '[]')[1:-1].split(',') if tag.strip()],
        "additional_details": {
            "problem": fields.get('problem', ''),
            "solution": fields.get('solution', ''),
            "benefits": fields.get('benefits', ''),
            "limitations": fields.get('limitations', ''),
            "references": [ref.strip() for ref in fields.get('references', '[]')[1:-1].split(',') if ref.strip()]
        }
    }
    
    return practice_data

async def create_practice(practice_data: dict) -> None:
    """Отправляет практику через API."""
    # Пропускаем уже созданные практики
    normalized_title = normalize_title(practice_data['title'])
    if normalized_title in successful_practices:
        logger.info(f"Skipping already created practice: {practice_data['title']}")
        return

    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"Starting creation of practice: {practice_data['title']}")
            response = await client.post(
                "http://localhost:8080/practice/create",
                json=practice_data,
                timeout=180.0
            )
            response.raise_for_status()
            logger.info(f"Successfully created practice: {practice_data['title']}")
            # Сохраняем детали ответа
            logger.info(f"Response details for {practice_data['title']}: {response.json()}")
            # Добавляем в список успешных
            with open(success_file, "a", encoding='utf-8') as f:
                f.write(f"{practice_data['title']}\n")
            successful_practices.add(normalized_title)
        except Exception as e:
            logger.error(f"Failed to create practice {practice_data['title']}: {e}")
            # Сохраняем данные неудачной практики
            with open(failed_practices_file, "a", encoding='utf-8') as f:
                f.write(f"{practice_data['title']}\n")

async def main():
    start_time = datetime.now()
    logger.info(f"Starting import process at {start_time}")
    logger.info(f"Already created practices: {len(successful_practices)}")
    
    # Путь к файлу с практиками
    md_file = Path("/home/minic/ai/aigents-docs/practice_examples/first 100.md")
    
    # Читаем содержимое файла
    content = md_file.read_text(encoding='utf-8')
    
    # Разделяем на отдельные практики
    practices = re.split(r'\n(?=### \d+\.)', content)
    
    # Статистика
    total = len([p for p in practices if p.strip()])
    success = len(successful_practices)
    failed = 0
    skipped = 0
    
    # Обрабатываем каждую практику
    for i, practice in enumerate(practices, 1):
        if not practice.strip():
            continue
            
        practice_data = await parse_practice(practice)
        if practice_data:
            if normalize_title(practice_data['title']) in successful_practices:
                skipped += 1
                logger.info(f"Skipping already created practice: {practice_data['title']}")
                continue

            try:
                await create_practice(practice_data)
                success += 1
            except Exception as e:
                failed += 1
                logger.error(f"Unexpected error processing practice {practice_data['title']}: {e}")
            
            # Пауза между запросами - 30 секунд
            if i < len(practices):  # Не ждем после последней практики
                logger.info(f"Waiting 30 seconds before next practice...")
                await asyncio.sleep(30)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Итоговая статистика
    summary = f"""
    Import completed at {end_time}
    Total duration: {duration}
    Total practices: {total}
    Successfully created: {success}
    Failed: {failed}
    Skipped (already existed): {skipped}
    Log file: {log_file}
    """
    logger.info(summary)

if __name__ == "__main__":
    asyncio.run(main()) 