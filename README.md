# ✨ OneClick Insights Generator
**A one-click Automated EDA web app that converts CSV files into decision-ready insights.** I built this to eliminate the slow, manual loop of loading data, profiling, plotting, and writing summaries. With one action, the app **profiles the dataset → generates visualizations → drafts a structured, executive-style report with LLM insights** and renders it in a clean, sectioned UI.

## 1) Problem identified, one-click solution, and user value
**The real-world pain:** As AI and data science grow inside teams, the first mile of analysis (getting from “here’s a CSV” to “here are the insights”) is still repetitive: fix encodings, scan missing values/outliers, try a few plots, and then write a narrative for stakeholders. This costs hours per dataset and produces inconsistent quality.
**My one-click solution:** I upload a CSV and press `[Generate]`. The app automatically:
- Profiles the data (missing values, outliers, correlations).
- Creates clean visualizations (scatter, histogram + density, correlation heatmap).
- Produces a structured README with **LLM-written insights** and next steps (≥5 per figure).
- Renders the report so each figure appears beside its insights (no duplication, nothing left out).
**Who benefits:** Analysts, PMs, founders, and domain experts who want **fast, repeatable, decision-oriented analysis** without writing code every time.

## 2) User persona, use cases, and expected outcomes
- **User persona**
  - Busy data analysts needing a rapid first pass on new datasets.
  - Product managers and founders who want quick, readable insights.
  - Domain specialists who are data-curious but not notebook-heavy.
- **Use cases**
  - First-look EDA for a newly received CSV (client export, experiment log, marketplace dump).
  - Quality checks: missing values, outlier counts, suspicious correlations.
  - Rapid briefing decks: charts + narrative insights for decision meetings.
- **Outcomes**
  - Minutes to a readable, shareable report.
  - Consistent baseline analysis across datasets.
  - Actionable, non-generic bullets that translate directly into next steps.

## 3) Skills & Tech Used

- **Tech stack:** `Python`, `Flask`, `Jinja2`, `Bootstrap`, `Font Awesome`, `JavaScript`
- **AI / LLM:**` OpenAI API (gpt-4o-mini)`, structured chat prompts, decision-grade data-to-text
- **Data & viz libs:** `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `joypy`
- **Web & security:** `Werkzeug` secure_filename, sessions
- **EDA methods:**` missing-value counts`, `IQR/z-score` outlier checks, `Pearson correlation matrix`
- **Visualization generation:** scatterplot , histogram + density (KDE), correlation heatmap

## 4) Pipeline details (step-by-step)
```
  1) Upload CSV (UI)
  └─ 2) Save to /uploads (secure_filename)
     └─ 3) Run analysis engine (subprocess)
        └─ 4) Detect encoding & Load DataFrame (pandas + chardet)
           └─ 5) Profile: Missing, Outliers, Correlations (Pearson)
              └─ 6) Graph Selection (LLM with rules)
                 └─ 7) Generate PNGs: Scatter, Hist+KDE, Heatmap
                    └─ 8) Draft README.md (LLM narrative, ≥5 bullets/figure)
                       └─ 9) Create dataset folder (CSV stem)
                          └─ 10) Write README.md (UTF-8) + PNGs
                             └─ 11) Report route
                                └─ 12) Parse README → Sections (readme_parser)
                                   └─ 13) Normalize Titles ↔ PNG Map
                                      └─ 14) Render: Figure + Insights side-by-side
```
- **Upload & validation:**
  - Single CSV via drag-and-drop or file picker.
  - Filename sanitized (`secure_filename`); stored under `/uploads/`; path kept in session.
  - Extension guard (`.csv`) and JSON status response.
- **Invocation pattern:** Analysis runs as a **subprocess** (`autolysis.py`) for isolation and easy CLI reuse.
- **Encoding & load:**
  - Raw bytes inspected; `chardet` suggests encoding if UTF-8 fails.
  - CSV loaded into `pandas` with safe dtype/NA handling.
- **Schema inference:** Numeric vs categorical separation; short dataset preview captured for prompts.
- **Profiling:**
  - `Missing-value counts` per column.
  - Outlier counts using `z-score/IQR` thresholds on numeric columns.
  - `Pearson correlation matrix` across numeric columns, later plotted as a heatmap.
- **Graph selection (LLM + rules):**
  - Prompt includes: column names, row/column counts, numeric/categorical lists, cardinality.
  - Rules enforce:
    - `Scatterplot` for two numeric columns with sufficient uniques; favored for larger row counts.
    - `Histogram + density` for continuous variables (≥ ~20 uniques).
    - `Heatmap` always included for correlations.
    - Non-redundancy and use of existing columns only.
- **Visualization generation:**
  - Scatterplot with sensible alpha, axis labels, and titles.
  - Histogram + KDE with appropriate bins (e.g., ~30).
  - Correlation heatmap via seaborn on the Pearson matrix.
  - All charts saved as `PNG` in the dataset-named folder.
- **Narrative generation (`README.md`):**
  - Instruction-first prompt requests **decision-grade bullets** (≥5/figure), key insights, implications, and a concise conclusion.
  - Subsections are forced for every generated figure, ensuring no PNG is left without insights.
  - Raw profiling (**“Preliminary Test Results”**) placed inside a collapsible `<details>` block.
- **Artifact writing:** `README` written as UTF-8; figures and README co-located under `/<dataset_name>/`.
- **Structured rendering (web):**
  - README read with encoding fallback; parsed into a **heading tree** (`custom readme_parser`).
  - Title↔image mapping normalizes text (lowercase, strip spaces/underscores/hyphens) so headings like “Correlation Matrix Heatmap” match `correlation_heatmap.png`.
- **UI presentation:**
  - **Figure + insights** rendered side-by-side via Bootstrap grid.
  - Duplicate sections suppressed; every PNG is paired with the relevant insights.
  - **Dark neon theme** and a loading overlay during report generation.

## 5) How to run
- **Prerequisites:**
  - Python 3.10+
  - OpenAI API key (`OPENAI_API_KEY`) available to the process

- **Create and activate a virtual environment:**
```
# macOS/Linux
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```
- **Provide environment variables:**
```\
# macOS/Linux
export OPENAI_API_KEY="YOUR_KEY"
export FLASK_SECRET_KEY="some-random-string"

# Windows (PowerShell)
$env:OPENAI_API_KEY="YOUR_KEY"
$env:FLASK_SECRET_KEY="some-random-string"
```
Or place them in a .env file at the project root:
```
OPENAI_API_KEY=YOUR_KEY
FLASK_SECRET_KEY=some-random-string
```
- **Run the app:**
```
python app.py
# or
flask run
```
Open `http://127.0.0.1:5000`, upload a `CSV`, and click `[Generate]`. A loading overlay appears during processing, followed by a report where each figure is displayed beside its insights.

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/074b4ae9-3ee5-46ca-b4bf-f1c2979e8b3d" />
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/d4ea82d5-1bfd-46e5-aed6-b2d6ec52643f" />
<img width="600" height="7200" alt="screencapture-127-0-0-1-5000-report-goodreads-2025-09-12-23_57_12" src="https://github.com/user-attachments/assets/d19c6c9a-0569-4804-87b0-def9dc2a4dd1" />
