#программирование #разработка #frontend #vibecode #webразработка 

Вот **упрощённая версия раздела «Контроль версий»** для маленького проекта с одной веткой `main`.  
Вы можете заменить соответствующий раздел в вашем `AGENTS.md` этим текстом.

---
## Version Control (simplified for a small project)

### 1. Branch
- The entire project lives in a GitHub repository.
- There is **only one main branch** – `main`.  
  All changes are committed directly to `main`.  
  If you are working on a large feature that might break things, you may create a **temporary branch** (e.g., `feature/new-screen`) and merge it back into `main` after completion. For most cases, committing straight to `main` is sufficient.

### 2. Versioning (tags)
- Releases are marked with **tags** in the format `vX.Y.Z` (e.g., `v1.2.3`).
- The version number follows **SemVer** rules (single version for the whole application – frontend + API):
  - **X (major)** – global, breaking changes (architecture overhaul, removal of key screens, API contract changes).
  - **Y (minor)** – new functionality (new screen, new fields, significant redesign).
  - **Z (patch)** – minor bug fixes, small UI tweaks, text corrections.

### 3. Commits
- All commits must follow the **[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)** specification.
- Format: `<type>(<scope>): <description>`
  - **Type**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`, `revert`.
  - **Scope** (optional): `frontend`, `api`, `certificate`, `styles`, etc.
  - **Description** – brief, in the imperative mood (e.g., “add”, “fix”).
- Examples:  
  `feat(frontend): add music selection screen`  
  `fix(api): fix date saving`  
  `docs: update README`

### 4. CHANGELOG
- A `CHANGELOG.md` file exists in the project root.
- For each release (tag), add a new entry with the changes in the **[Keep a Changelog](https://keepachangelog.com/en/1.0.0/)** format:
  - `Added` – new features
  - `Changed` – changes to existing functionality
  - `Deprecated` – soon‑to‑be removed features
  - `Removed` – removed features
  - `Fixed` – bug fixes
  - `Security` – security improvements

### 5. Creating a release
When enough changes have accumulated and everything is tested:
1. Make sure all necessary commits are already in `main`.
2. Create a tag with the new version:
.  ```bash
   git tag -a v1.0.0 -m "Release 1.0.0"
   git push origin v1.0.0
3. (Optional) CI will automatically build production artefacts.
### 6. CI/CD (basic)
Set up GitHub Actions (or another CI service) to run automated checks on **pushes to `main`** and **tag creation**:
- Build both sub‑projects (`npm run build`) – ensure the code compiles.
- (In the future) run tests.    
- Check formatting (Prettier) and linting (ESLint) – add these tools to the project.    
### 7. Dependencies
- `package-lock.json` (or `yarn.lock`) **must** be committed – this guarantees identical package versions across all developers and servers.    
- Update dependencies regularly, but with caution (test the application after updating).    

### 8. Documentation
- In `README.md`, briefly describe the branch and commit workflow (you can link to this section).    
- This section in `AGENTS.md` serves as the main guideline for everyone contributing to the project.