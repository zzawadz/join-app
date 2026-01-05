# Record Linkage Web Application

A comprehensive web application for record linkage supporting deduplication, Fellegi-Sunter probabilistic model, machine learning classification, and active learning.

## Features

- **Deduplication**: Find duplicate records within a single dataset
- **Record Linkage**: Link records between two different datasets
- **Fellegi-Sunter Model**: Probabilistic record linkage using the classic FS model
- **Machine Learning**: Train classifiers (Logistic Regression, Random Forest) on labeled pairs
- **Active Learning**: Interactive labeling UI with uncertainty sampling
- **CSV Upload**: Upload and manage datasets with automatic encoding detection
- **Column Mapping**: Auto-suggest column mappings based on column names and content
- **Fuzzy Matching**: Jaro-Winkler, Levenshtein, Soundex, and other string distance metrics
- **Blocking Strategies**: Standard blocking, sorted neighborhood, phonetic blocking
- **Team Collaboration**: Organizations with member roles

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML + HTMX + Tailwind CSS
- **Database**: SQLite (easily upgradable to PostgreSQL)
- **ML Libraries**: scikit-learn, jellyfish

## Quick Start

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create environment file**:
   ```bash
   cp .env.example .env
   # Edit .env and set a secure SECRET_KEY
   ```

4. **Run the application**:
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Open in browser**: http://localhost:8000

## Project Structure

```
join-app/
├── app/
│   ├── api/                  # API endpoints
│   │   ├── auth.py           # Authentication
│   │   ├── datasets.py       # Dataset management
│   │   ├── projects.py       # Project CRUD
│   │   ├── linkage.py        # Linkage execution
│   │   ├── models.py         # ML model training
│   │   └── labeling.py       # Active learning
│   ├── core/
│   │   ├── security.py       # JWT, password hashing
│   │   └── linkage/
│   │       ├── comparators.py    # String similarity functions
│   │       ├── blocking.py       # Blocking strategies
│   │       ├── fellegi_sunter.py # FS model
│   │       ├── ml_classifier.py  # ML classifiers
│   │       └── active_learning.py
│   ├── db/
│   │   ├── database.py       # SQLAlchemy setup
│   │   └── models.py         # ORM models
│   ├── schemas/              # Pydantic schemas
│   ├── services/             # Business logic
│   │   ├── csv_processor.py  # CSV parsing
│   │   ├── column_mapper.py  # Column mapping suggestions
│   │   └── storage.py        # File storage
│   ├── templates/            # Jinja2 HTML templates
│   ├── config.py
│   └── main.py               # FastAPI app
├── static/                   # CSS, JS
├── storage/                  # Uploaded files, models
├── requirements.txt
└── README.md
```

## Usage Workflow

1. **Register/Login**: Create an account to get started
2. **Create Project**: Choose between Deduplication or Linkage
3. **Upload Datasets**: Upload CSV files for source and target (or single for dedup)
4. **Map Columns**: Configure which columns to compare
5. **Configure Comparison**: Select comparison methods (Jaro-Winkler, Levenshtein, etc.)
6. **Set Blocking**: Choose blocking strategy to reduce comparison space
7. **Label Pairs** (Optional): Use active learning to label pairs and train a model
8. **Run Linkage**: Execute the linkage job
9. **Export Results**: Download matched pairs as CSV

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Comparison Methods

| Method | Description | Best For |
|--------|-------------|----------|
| Exact | Case-insensitive exact match | IDs, codes |
| Jaro-Winkler | Edit distance optimized for typos | Names |
| Levenshtein | General edit distance | Addresses |
| Soundex | Phonetic matching | English names |
| Numeric | Numeric similarity with tolerance | Ages, amounts |
| Date | Date comparison with day tolerance | Birthdates |

## Blocking Strategies

| Strategy | Description |
|----------|-------------|
| Standard | Only compare records with identical blocking keys |
| Sorted Neighborhood | Sort by key, compare within sliding window |
| Phonetic | Block by Soundex of name fields |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| SECRET_KEY | JWT signing key | dev-secret-key |
| DATABASE_URL | Database connection string | sqlite:///./storage/app.db |
| UPLOAD_DIR | Directory for uploaded files | storage/uploads |
| MODELS_DIR | Directory for ML models | storage/models |
| MAX_UPLOAD_SIZE_MB | Max upload size in MB | 100 |

## License

MIT
