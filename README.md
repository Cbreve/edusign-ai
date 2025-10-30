# EduSign-AI

This repository contains extraction utilities and datasets for the Harmonized GSL Dictionary.

## Processed outputs
- `backend/app/data/processed/gsl_dictionary.json`
- `backend/app/data/processed/gsl_dictionary.csv`

## Raw data (not stored in Git)
Large raw assets (PDFs, extracted images) are intentionally not tracked in Git to keep the repository lightweight and within platform limits.

- Place the raw PDF at `backend/app/data/raw/Harmonized_GSL_Dictionary_v3_2023.pdf` (or `Harmonized_GSL_Dictionary.pdf`).
- The `.gitignore` prevents committing large PDFs and `sign_images/`.
- If the raw file is needed by multiple contributors, share it via external storage (e.g., Drive/S3) and download locally before running the extractor.

## Running the extractor
```
python backend/app/utils/extract_gsl_dictionary.py
```
This generates the processed JSON/CSV and prints a brief summary.

## Data quality checks
- The extractor applies noise filters, column-aware parsing, optional OCR fallback with confidence gating, and adds metadata per entry.
- A small gold set lives at `backend/app/data/gold/gsl_gold.json`. The CI workflow validates extractor output against this set and fails when any item is missing.

## CI
GitHub Actions workflow (`.github/workflows/quality.yml`) installs dependencies, runs the extractor, and validates against the gold set on every push/PR to `main`.
