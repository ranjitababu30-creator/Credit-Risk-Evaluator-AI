# Contributing

## Development setup

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Train local model artifacts with `python -m src.train_model`.
4. Run the test suite with `python -m pytest`.

## Pull requests

- Keep changes focused and explain user-visible behavior.
- Add or update tests for changed behavior.
- Do not commit personal, financial, credential, or production applicant data.
- Do not commit generated model artifacts or SQLite databases; they are ignored by design.
- Keep the responsible-lending disclaimers intact.