#программирование #разработка #backend #AI/ML #OCR #vibecode

---
## Version Control (simplified for Unlimited-OCR project)

### 1. Repository & Branch
- The entire project lives in a GitHub repository: **https://github.com/viktorbrojs-sys/unlimited-ocr**
- There is **only one main branch** – `main`.  
  All changes are committed directly to `main`.  
  If you are working on a large feature that might break things, you may create a **temporary branch** (e.g., `feature/batch-processing`) and merge it back into `main` after completion. For most cases, committing straight to `main` is sufficient.

### 2. Versioning (tags)
- Releases are marked with **tags** in the format `vX.Y.Z` (e.g., `v1.0.0`).
- The version number follows **SemVer** rules (single version for the whole application):
  - **X (major)** – global, breaking changes (architecture overhaul, removal of key features, API contract changes).
  - **Y (minor)** – new functionality (new screens, new export formats, significant UI redesign).
  - **Z (patch)** – minor bug fixes, small UI tweaks, text corrections, documentation updates.

### 3. Commits
- All commits must follow the **[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)** specification.
- Format: `<type>(<scope>): <description>`
  - **Type**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`, `revert`.
  - **Scope** (optional): `frontend`, `backend`, `model`, `docs`, `install`, `pdf`, `api`, etc.
  - **Description** – brief, in the imperative mood (e.g., "add", "fix").
- Examples:  
  `feat(backend): add batch file upload endpoint`  
  `fix(model): handle CUDA OOM gracefully`  
  `docs: update PROJECT_DOCS with roadmap`  
  `perf(pdf): reduce memory footprint for large PDFs`

### 4. Documentation Files
- `README.md` — краткое руководство для пользователей (быстрый старт)
- `PROJECT_DOCS.md` — полная документация проекта (архитектура, технические детали, планы развития)
- `CHANGELOG.md` — история изменений по версиям (Keep a Changelog формат)
- `AGENTS.md` — этот файл, инструкция для AI-агентов и разработчиков

### 5. CHANGELOG
- A `CHANGELOG.md` file exists in the project root.
- For each release (tag), add a new entry with the changes in the **[Keep a Changelog](https://keepachangelog.com/en/1.0.0/)** format:
  - `Added` – new features
  - `Changed` – changes to existing functionality
  - `Deprecated` – soon‑to‑be removed features
  - `Removed` – removed features
  - `Fixed` – bug fixes
  - `Security` – security improvements

### 6. Creating a release
When enough changes have accumulated and everything is tested:
1. Make sure all necessary commits are already in `main`.
2. Update `CHANGELOG.md` with the new version entries.
3. Create a tag with the new version:
   ```bash
   git tag -a v1.0.0 -m "Release 1.0.0"
   git push origin v1.0.0
   ```

### 7. Dependencies
- `requirements.txt` **must** be committed — this guarantees identical package versions across all developers and servers.
- Update dependencies regularly, but with caution (test the application after updating).
- Pay attention to version conflicts (see PROJECT_DOCS.md for known conflicts between gradio 6 and transformers 4.57.1).

### 8. Documentation & Memory
- In `README.md`, briefly describe the branch and commit workflow (you can link to this section).
- This section in `AGENTS.md` serves as the main guideline for everyone contributing to the project.
- **Always read `PROJECT_DOCS.md` before making significant changes** to understand the architecture, technical decisions, and development plans.
- `PROJECT_DOCS.md` contains the project "memory" — what was done, what is being done, and what is planned. This is critical for AI agents to maintain context across sessions.

### 9. Working Without GitHub Access
- If you cannot access the GitHub repository directly, work locally in the `/workspace` directory.
- All changes should still follow the commit message format and be documented in `CHANGELOG.md`.
- When GitHub access is restored, sync your local changes with the remote repository.

---

*Last updated: August 2024*  
*Document version: 1.0.0*
