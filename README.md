# EduSign AI

Real-time speech-to-sign-language translation platform for the deaf community.

## Project Structure

```
EduSign-AI/
├── backend/          # FastAPI backend (AI & API)
├── frontend/         # Next.js frontend
├── integrations/     # SDKs for Zoom, Google Meet, Teams
├── scripts/          # Data processing and utility scripts
└── docs/             # Documentation and guides
```

## Quick Start

### Backend (Python/FastAPI)
```bash
cd backend
pip install -r ../requirements.txt
uvicorn app.main:app --reload
```

### Frontend (Next.js)
```bash
npm install
npm run dev
```

## Dataset Management

### Extracting Sign Images from GSL Dictionary PDF
```bash
python scripts/extract_gsl_sign_images.py
```

### Extracting Dictionary Data
```bash
python scripts/extract_gsl_dictionary.py
```

### Quality Assurance
```bash
python scripts/dataset_quality_assurance.py
python scripts/validate_sign_images.py
python scripts/validate_video_frames.py
```

### YouTube Video Processing
```bash
# Download videos
python scripts/download_youtube_sign_videos.py

# Extract frames
python scripts/extract_frames_from_videos.py

# Validate frames
python scripts/validate_video_frames.py
```

## Data Structure

- **Raw Data**: `backend/app/data/raw/`
  - `sign_images/` - Extracted sign images from PDF
  - `youtube_videos/` - Downloaded YouTube videos
  - `video_frames/` - Extracted video frames
  - `Harmonized_GSL_Dictionary.pdf`

- **Processed Data**: `backend/app/data/processed/`
  - `gsl_dictionary.json` - Dictionary entries
  - `validated_frames/` - Quality-validated frames

## Documentation

See `docs/` for detailed guides:
- `DATASET_QUALITY_GUIDE.md` - Dataset quality best practices
- `YOUTUBE_DATASET_GUIDE.md` - YouTube video processing guide
- `api_documentation.md` - Backend API reference

## Note on Large Files

Large raw assets (PDFs, videos, images) are intentionally not tracked in Git (see `.gitignore`). Share via external storage if needed.
