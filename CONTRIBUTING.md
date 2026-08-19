# Contributing

Руководство для тех, кто дорабатывает проект — человек или AI-агент.

## Перед началом работы

1. Прочитайте [`README.md`](README.md) — быстрый старт и что вообще делает проект.
2. Прочитайте [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — как устроен `app.py`,
   какие есть блокировки и почему, поток данных.
3. Посмотрите [`ROADMAP.md`](ROADMAP.md) — что уже сделано и что запланировано,
   чтобы не дублировать работу.
4. Перед коммитом обязательно прогоните:
   ```bash
   python3 -m py_compile app.py
   bash -n start.sh && sh -n install.sh
   ```
   Это же проверяет CI (`.github/workflows/syntax-check.yml`) — но лучше
   поймать ошибку локально, чем ждать красный крестик в PR.

## Ветвление

- Основная ветка — `main`. Небольшие изменения коммитятся напрямую в неё.
- Для крупных фич — временная ветка `feature/<name>`, затем merge в `main`.

## Коммиты

Используем [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>(<scope>): <description>
```

- **type**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`, `revert`
- **scope** (опционально): `backend`, `frontend`, `model`, `docs`, `install`, `pdf`, `api`
- **description** — коротко, в повелительном наклонении

Примеры:
```
feat(backend): add batch file upload endpoint
fix(model): handle CUDA OOM gracefully
docs: update installation instructions
perf(pdf): reduce memory footprint for large PDFs
```

Для мультистрочных коммитов первая строка — краткое summary, ниже —
пояснение *почему*, а не только *что* (см. историю коммитов `git log`
для примера — это лучше, чем пересказывать диффы).

## Версионирование

[SemVer](https://semver.org/lang/ru/): `vX.Y.Z`

- **X** — breaking changes (архитектура, изменение API-контракта эндпоинтов)
- **Y** — новая функциональность
- **Z** — багфиксы, мелкие правки, обновление документации

Текущая версия лежит в файле [`VERSION`](VERSION) и дублируется строкой
`id="app-version"` в `index.html` (строка с `<span id="app-version">`) —
при бампе версии обновляйте оба места.

## Релиз

1. Убедитесь, что все нужные коммиты в `main`.
2. Добавьте запись в [`CHANGELOG.md`](CHANGELOG.md) по формату
   [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/)
   (`Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security`).
3. Обновите `VERSION` и версию в `index.html`.
4. Тег и пуш:
   ```bash
   git tag -a v1.2.1 -m "Release 1.2.1"
   git push origin v1.2.1
   ```

## Зависимости

- `requirements.txt` коммитится всегда — гарантирует одинаковые версии
  пакетов у всех.
- `transformers==4.57.1` ставится отдельным шагом **после**
  `requirements.txt` (см. [`install.sh`](install.sh)) — иначе pip не может
  разрешить конфликт `huggingface-hub` между gradio 6 и transformers.
  Если меняете версию gradio или transformers, перепроверьте, актуален ли
  ещё этот конфликт, и обновите README/install.sh/start.sh согласованно.
- Не добавляйте `bitsandbytes`/`accelerate` обратно без явной причины —
  они не совместимы с архитектурой модели (MoE + R-SWA), это уже
  проверено и задокументировано в `docs/ARCHITECTURE.md`.

## Работа без доступа к GitHub

Если агент/среда временно не имеет доступа к удалённому репозиторию —
работайте в локальной копии как обычно, коммитьте с нормальными
сообщениями, и синхронизируйтесь (`git push`), как только доступ
восстановится. Не создавайте отдельные "инструкции по миграции" в
корне репозитория для разовых ситуаций — это создаёт путаницу для
следующего разработчика; если нужна заметка "для себя", держите её вне
репозитория.
