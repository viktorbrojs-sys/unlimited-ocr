# Unlimited-OCR Project Documentation

## Project Overview

**Unlimited-OCR** — это локальная версия демо [baidu/Unlimited-OCR](https://huggingface.co/spaces/baidu/Unlimited-OCR), которая позволяет запускать распознавание текста (OCR) на вашей собственной машине с NVIDIA GPU вместо использования Hugging Face ZeroGPU.

### Ключевые возможности

- **Модель**: `baidu/Unlimited-OCR` (построена на DeepSeek-OCR)
- **Назначение**: «one-shot long-horizon parsing» — полное распознавание длинных документов и многостраничных PDF за один проход
- **Интерфейс**: Тот же интерфейс, что и в оригинальном Space (index.html из Space)
- **Стриминг**: Потоковая передача распознанного текста в реальном времени
- **Поддержка PDF**: Автоматическая конвертация PDF в изображения постранично

---

## Version History & Changelog

### v1.0.0 — Initial Local Deployment (2024)

**Дата**: Август 2024

**Added**:
- Локальный сервер на базе Gradio 6 с FastAPI endpoints
- Стриминг OCR вывода через SSE (Server-Sent Events)
- Поддержка загрузки модели с выбором варианта (bf16 CUDA / CPU float32)
- Автоматический fallback на квантованные режимы при нехватке VRAM
- Конвертация PDF в PNG через PyMuPDF (fitz)
- Фронтенд из оригинального HuggingFace Space без изменений
- Скрипт установки `install.sh` для создания виртуального окружения

**Changed**:
- Убран декоратор `@spaces.GPU` — теперь модель работает локально без ограничений по времени
- Модель загружается по требованию из веб-интерфейса, а не при старте сервера
- Блокировка `_infer_lock` для сериализации вызовов модели (один инференс за раз)

**Technical Decisions**:
- Двухшаговая установка зависимостей (сначала requirements.txt, потом transformers==4.57.1)
- Причина: конфликт версий huggingface-hub между gradio 6 и transformers 4.57.1
- Перехват sys.stdout для стриминга токенов из модели в браузер
- Автоматический перезапуск сервера с `CUDA_VISIBLE_DEVICES=""` для CPU режима

---

## Architecture

### Компоненты системы

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (index.html)                    │
│  - PDF.js для превью PDF                                    │
│  - Терминальный вывод с потоковым текстом                   │
│  - Выбор режима: LONG (gundam, 640px) / BASE (1024px)       │
└─────────────────────────────────────────────────────────────┘
                            ↓↑ HTTP/SSE
┌─────────────────────────────────────────────────────────────┐
│                    Backend (app.py)                          │
│  - Gradio 6 Server с FastAPI endpoints                      │
│  - /run_ocr — стриминг OCR для одного изображения           │
│  - /explode_pdf — конвертация PDF → PNG                     │
│  - /load_model — загрузка модели по требованию              │
│  - model_status — статус загруженной модели                 │
└─────────────────────────────────────────────────────────────┘
                            ↓↑ Python API
┌─────────────────────────────────────────────────────────────┐
│              Model Layer (transformers + torch)              │
│  - AutoModel.from_pretrained("baidu/Unlimited-OCR")         │
│  - tokenizer = AutoTokenizer.from_pretrained(...)           │
│  - infer() метод модели с параметрами crop_mode, ngram и т.д.│
└─────────────────────────────────────────────────────────────┘
```

### Поток данных

1. **Загрузка модели**:
   - Пользователь выбирает вариант (auto/bf16/cpu) в UI
   - Вызывается `/load_model` endpoint
   - Модель загружается в отдельном потоке
   - Прогресс стримится обратно в UI

2. **Обработка изображения**:
   - Пользователь загружает изображение или PDF
   - Для PDF: вызывается `/explode_pdf` → список PNG
   - Для каждого PNG: вызывается `/run_ocr`
   - Модель.infer() пишет в stdout → перехватывается → стримится клиенту
   - Результат сохраняется во временную папку `ocr_out_*`

3. **Стриминг**:
   - `@app.api(stream_every=0.1)` декоратор
   - Каждый yielded dict отправляется через SSE
   - Формат: `{"text": str, "done": bool}`

---

## File Structure

| Файл | Назначение |
|------|------------|
| `app.py` | Сервер: gradio 6 `Server`, эндпоинты `/run_ocr`, `/explode_pdf`, `/load_model`, отдаёт `index.html` |
| `index.html` | Фронтенд из Space (PDF.js-превью + терминальный вывод), без изменений |
| `install.sh` | Создание `.venv` и двухшаговая установка зависимостей |
| `requirements.txt` | Зависимости: torch, gradio>=6.0, transformers, PyMuPDF, fastapi, uvicorn |
| `README.md` | Краткое руководство пользователя |
| `AGENTS.md` | Инструкция для AI-агентов и разработчиков |
| `CHANGELOG.md` | История изменений (этот файл) |
| `PROJECT_DOCS.md` | Полная документация проекта (этот файл) |

---

## Technical Requirements

### Аппаратные требования

- **OS**: Linux
- **GPU**: NVIDIA с ≥16 ГБ VRAM (рекомендуется)
- **Драйвер**: Свежий драйвер NVIDIA (≥575 для CUDA 12.9)
- **RAM**: ~10 ГБ свободного места (torch + веса модели)
- **CPU Mode**: Работает, но очень медленно

### Программные требования

- **Python**: 3.12
- **CUDA**: 12.9 (через torch сборку по умолчанию)
- **Зависимости**:
  ```
  torch (с CUDA support)
  gradio>=6.0
  transformers==4.57.1
  PyMuPDF (fitz)
  fastapi
  uvicorn
  huggingface-hub
  ```

### Конфликты зависимостей

**Проблема**: transformers 4.57.1 требует `huggingface-hub<1.0`, а gradio 6 — `>=1.16`

**Решение**: Двухшаговая установка
```bash
# Шаг 1: основные зависимости
pip install -r requirements.txt

# Шаг 2: transformers (переписывает huggingface-hub)
pip install transformers==4.57.1
```

**Предупреждение pip**: После установки pip предупредит о конфликте с gradio — это ожидаемо, сервер работает корректно.

---

## Usage Guide

### Установка

```bash
./install.sh
```

Скрипт создаёт `.venv` и устанавливает зависимости в два шага.

### Запуск

```bash
.venv/bin/python app.py
```

Дождитесь в консоли `Model ready on cuda` (при первом запуске — после загрузки весов), затем откройте **http://127.0.0.1:7860**.

**Переменные окружения**:
- `HOST` — адрес сервера (по умолчанию `127.0.0.1`)
- `PORT` — порт (по умолчанию `7860`)
- `UNLIMITED_OCR_DEVICE=cpu` — принудительный CPU режим

### Использование

1. Перетащите картинку или PDF в зону загрузки
2. PDF отрисуется постранично в превью
3. Выберите режим:
   - **LONG** (gundam) — быстрый, кроп 640 px
   - **BASE** — точнее, кроп 1024 px
4. Prompt по умолчанию: `document parsing.` (можно менять)
5. Нажмите START — текст появляется потоком
6. COPY — скопировать результат

### Режимы модели

| Вариант | Описание | Требования |
|---------|----------|------------|
| `auto` | Попытка bf16 CUDA, fallback на CPU | ≥16 ГБ VRAM для bf16 |
| `bf16` | CUDA bfloat16 | ≥16 ГБ VRAM, драйвер ≥575 |
| `cpu` | CPU float32 | Любой CPU, очень медленно |

**Квантование**: Bitsandbytes (8/4-bit) НЕ совместим с архитектурой MoE + R-SWA этой модели.

---

## Known Issues & Workarounds

### CUDA Out of Memory

**Симптом**: Карта на 8 ГБ, bf16-веса (~6.7 ГБ) не помещаются вместе с CUDA-контекстом

**Решение**:
1. Установите `bitsandbytes accelerate`:
   ```bash
   .venv/bin/pip install bitsandbytes accelerate
   ```
2. Код автоматически попытается: bf16 → 8-бит → 4-бит
3. 8-бит почти не теряет качество OCR

### Expected all tensors to be on the same device

**Причина**: Модель загрузилась в CPU при видимом CUDA (старая версия app.py)

**Решение**:
```bash
git pull
.venv/bin/pip install bitsandbytes accelerate
```

### CUDA driver version is insufficient

**Причина**: Драйвер старее CUDA-сборки torch

**Решение**:
1. Поставьте свежий драйвер NVIDIA
2. Или нужную сборку torch с https://download.pytorch.org:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```

### pip ругается на huggingface-hub

**Это ожидаемо** — см. раздел "Конфликты зависимостей" выше.

### Результаты сохраняются в临时ную папку

Выходные файлы (.txt/.md) сохраняются в:
```
/tmp/ocr_out_*/  # путь виден в консоли сервера
```

---

## Development Notes

### Структура кода app.py

**Глобальное состояние**:
- `tokenizer` — кэшированный токоенизатор
- `model` — кэшированная модель
- `_model_label` — метка текущего варианта модели
- `_infer_lock` — блокировка для сериализации инференса
- `_model_state_lock` — блокировка для проверки состояния модели
- `_temp_dirs` — список временных директорий для очистки

**Основные функции**:

1. `_cleanup()` — очистка при выходе (модель, temp dirs)
2. `_variant_kwargs(name)` — kwargs для from_pretrained по названию варианта
3. `_load_model_sync(variant, q)` — загрузка модели с прогрессом в queue
4. `_target_labels(variant)` — список меток, которые попробует вариант
5. `load_model(variant)` — API endpoint для загрузки модели
6. `model_status()` — API endpoint для статуса модели
7. `pdf_to_images(pdf_path, dpi)` — конвертация PDF → PNG
8. `_collect_output(out_dir)` — чтение выходных файлов модели
9. `run_ocr(image_path, mode, prompt)` — основной OCR endpoint со стримингом
10. `explode_pdf(pdf_file)` — API endpoint для конвертации PDF
11. `homepage()` — отдача index.html

### ThreadTargetedStdout

Критический компонент для стриминга:
- Патчит sys.stdout только для целевого потока инференса
- Фильтрует TPS-сообщения ("tps:", "tokens/s")
- Отправляет текст в queue для стриминга через SSE

### Автоматический перезапуск для CPU режима

Для CPU режима сервер автоматически перезапускается с `CUDA_VISIBLE_DEVICES=""`:
```python
os.execve(sys.executable, [sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]], env)
```

Причина: model.infer() отправляет тензоры на cuda, если torch.cuda.is_available(), даже если модель на CPU.

---

## Future Plans & Roadmap

### Приоритет 1: Стабильность и производительность

- [ ] Добавить обработку ошибок для PDF с повреждёнными страницами
- [ ] Оптимизация памяти: явный gc.collect() между страницами PDF
- [ ] Добавить прогресс-бар для многостраничных PDF
- [ ] Кэширование результатов OCR для повторных запросов

### Приоритет 2: Функциональность

- [ ] Поддержка пакетной обработки нескольких файлов
- [ ] Экспорт результатов в различные форматы (DOCX, TXT, MD, JSON)
- [ ] Настройка параметров инференса через UI:
  - max_length
  - no_repeat_ngram_size
  - ngram_window
  - base_size / image_size
- [ ] Сохранение истории OCR сессий
- [ ] Поддержка drag-and-drop нескольких файлов одновременно

### Приоритет 3: Интеграции

- [ ] REST API для внешнего использования (не только через UI)
- [ ] Dockerfile для контейнеризации
- [ ] docker-compose.yml для развёртывания
- [ ] Интеграция с облачными хранилищами (S3, Google Drive)
- [ ] Webhook уведомления после завершения OCR

### Приоритет 4: Мониторинг и логирование

- [ ] Логирование запросов и ответов в файл
- [ ] Метрики производительности (время обработки, размер файла, длина текста)
- [ ] Health check endpoint для мониторинга
- [ ] Prometheus metrics export

### Приоритет 5: Документация и тесты

- [ ] Unit tests для ключевых функций
- [ ] Integration tests для полного пайплайна OCR
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Примеры использования в production
- [ ] Benchmark разных GPU и режимов

### Долгосрочные планы

- [ ] Поддержка других OCR моделей (EasyOCR, Tesseract, PaddleOCR)
- [ ] Сравнение результатов между моделями
- [ ] Fine-tuning на специфичных доменах (медицина, юриспруденция)
- [ ] Распознавание рукописного текста
- [ ] Поддержка дополнительных языков
- [ ] GUI приложение (Electron/Tauri) для десктопа

---

## Contributing Guidelines

### Ветвление

- Основная ветка: `main`
- Для больших фич: временные ветки `feature/<name>`
- Все изменения коммитятся напрямую в `main` для небольших правок

### Коммиты

Следуйте спецификации [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>
```

**Типы**:
- `feat` — новая функциональность
- `fix` — исправление багов
- `docs` — документация
- `style` — форматирование, отступы
- `refactor` — рефакторинг без изменений функциональности
- `perf` — улучшение производительности
- `test` — тесты
- `chore` — вспомогательные задачи

**Примеры**:
```
feat(frontend): add batch file upload
fix(backend): handle corrupted PDF pages
docs: update installation instructions
perf(model): reduce memory footprint for large images
```

### Версионирование

SemVer (Semantic Versioning): `vX.Y.Z`

- **X (major)** — breaking changes (архитектура, API contract)
- **Y (minor)** — новая функциональность
- **Z (patch)** — багфиксы, мелкие улучшения

### Создание релиза

```bash
# Убедитесь, что все коммиты в main
git tag -a v1.0.0 -m "Release 1.0.0"
git push origin v1.0.0
```

---

## Contact & Support

**Оригинальный проект**: [baidu/Unlimited-OCR на HuggingFace](https://huggingface.co/spaces/baidu/Unlimited-OCR)

**Модель**: [baidu/Unlimited-OCR на HuggingFace](https://huggingface.co/baidu/Unlimited-OCR)

**Лицензия**: Проверьте лицензию оригинальной модели и кода Space.

---

## Appendix: Environment Variables Reference

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `HOST` | `127.0.0.1` | Адрес сервера |
| `PORT` | `7860` | Порт сервера |
| `UNLIMITED_OCR_DEVICE` | — | `cpu` для принудительного CPU режима |
| `UNLIMITED_OCR_CPU_REEXEC` | — | Внутренняя, для перезапуска в CPU режиме |
| `CUDA_VISIBLE_DEVICES` | — | Управление видимостью GPU (для CPU режима) |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | Конфигурация аллокатора PyTorch |

---

*Последнее обновление: Август 2024*  
*Версия документа: 1.0.0*
