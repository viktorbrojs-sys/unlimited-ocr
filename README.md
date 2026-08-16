# Unlimited-OCR — локальный инференс

Локальная версия демо [baidu/Unlimited-OCR](https://huggingface.co/spaces/baidu/Unlimited-OCR):
тот же интерфейс (`index.html` из Space) и тот же стриминг распознанного текста,
но запуск на вашей машине с NVIDIA GPU вместо Hugging Face ZeroGPU.

Модель — `baidu/Unlimited-OCR` (построена на DeepSeek-OCR): «one-shot long-horizon
parsing» — полное распознавание длинных документов и многостраничных PDF за один проход.

## Требования

- Linux + NVIDIA GPU (≥ 16 ГБ VRAM рекомендуется), свежий драйвер NVIDIA
  (сборка torch по умолчанию использует CUDA 12.9 — драйвер ≥ 575)
- Python 3.12, ~10 ГБ свободного места (torch + веса модели)
- Первый запуск скачает веса модели (~7 ГБ) с Hugging Face

## Установка

```bash
./install.sh
```

Скрипт создаёт `.venv` и ставит зависимости в два шага: сначала
`requirements.txt` (torch, gradio 6 и пр.), затем `transformers==4.57.1`.
Второй шаг нужен потому, что transformers 4.57.1 требует `huggingface-hub<1.0`,
а gradio 6 — `>=1.16`; вместе за один resolve pip их не поставить. После
установки pip предупредит о конфликте с gradio — это ожидаемо, сервер работает
(ровно так же это обходится в самом Space).

## Запуск

```bash
.venv/bin/python app.py
```

Дождитесь в консоли `Model ready on cuda` (при первом запуске — после загрузки
весов), затем откройте **http://127.0.0.1:7860**.

- Адрес/порт: переменные окружения `HOST` / `PORT` (по умолчанию `127.0.0.1:7860`).
- Без CUDA сервер запустится на CPU в float32 — работать будет, но очень медленно.

## Использование

Как в Space:

1. Перетащите картинку или PDF в зону загрузки (PDF отрисуется постранично).
2. Режим: **LONG** — быстрый (gundam, кроп 640 px), **BASE** — точнее (1024 px).
3. Prompt по умолчанию — `document parsing.` (можно менять).
4. START — текст появляется потоком, постранично для PDF; COPY — скопировать результат.

## Структура

| Файл | Назначение |
|---|---|
| `app.py` | Сервер: gradio 6 `Server`, эндпоинты `/run_ocr` (стриминг) и `/explode_pdf` (PDF → PNG), отдаёт `index.html` |
| `index.html` | Фронтенд из Space (PDF.js-превью + терминальный вывод), без изменений |
| `install.sh` | Создание `.venv` и двухшаговая установка зависимостей |

## Возможные проблемы

- **CUDA out of memory при загрузке** (карты на 8 ГБ): bf16-веса (~6.7 ГБ)
  не помещаются вместе с CUDA-контекстом и активациями. `app.py` сам пытается
  по цепочке: bf16 → 8-бит → 4-бит → CPU. Для квантованных режимов один раз
  установите `.venv/bin/pip install bitsandbytes` (8-бит почти не теряет
  качество OCR). Принудительный CPU: `UNLIMITED_OCR_DEVICE=cpu .venv/bin/python app.py`.- **CUDA driver version is insufficient** — драйвер старее CUDA-сборки torch.
  Поставьте свежий драйвер либо нужную сборку torch с
  <https://download.pytorch.org> (например `--index-url https://download.pytorch.org/whl/cu126`).
- **pip ругается на huggingface-hub при установке transformers** — ожидаемо, см. выше.
- Результаты инференса также сохраняются в `.txt`/`.md` во временной папке
  `ocr_out_*` (путь виден в консоли сервера).
