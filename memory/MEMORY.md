# Memory Index

- [User profile: Russian, CPU-only PC](user-profile-russian-cpu-pc.md) — Russian language; GPU workloads run on a different machine
- [Prefers exact reference replicas](user-prefers-exact-reference-replica.md) — port the reference's own UI, not a simplified version
- [Prefers minimal-token solutions](user-prefers-minimal-token-solutions.md) — offer cost tiers, lead with cheapest that meets the need; partial coverage OK
- [unlimited-ocr local port status](unlimited-ocr-local-port.md) — квантование 8/4-bit несовместимо с моделью (убрано, commit 0a4f1c8); UI: auto/bf16/CPU с авто-перезапуском; ждём тест bf16 (освободить VRAM) или CPU на GPU-ПК; качество кириллицы не проверено
- [GitHub CLI ready](github-cli-setup.md) — authed as viktorbrojs-sys (SSH), git identity set; ask visibility before creating repos
- [ZCode session python is AppImage](zcode-session-python-is-appimage.md) — no pip in-session; verify deps via metadata, defer installs
- [ZCode Desktop RU localization — отложена](zcode-desktop-ru-localization-deferred.md) — цель: оболочка приложения (asar), не Chromium-диалоги; смещение squashfs 188392, план патча готов
