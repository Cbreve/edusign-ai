# YouTube Video Dataset Creation Guide

This guide explains how to create a high-quality sign language dataset from YouTube videos.

## Overview

The pipeline consists of:
1. **Download** videos from YouTube
2. **Extract** high-quality frames
3. **Validate** frame quality
4. **Integrate** with existing dictionary dataset

## Prerequisites

Install required dependencies:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install yt-dlp opencv-python numpy Pillow
```

## Step 1: Download YouTube Videos

### Download Single Video
```bash
python3 scripts/download_youtube_sign_videos.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Download Playlist
```bash
python3 scripts/download_youtube_sign_videos.py "https://www.youtube.com/playlist?list=PLAYLIST_ID" --playlist
```

### Options
- `--quality best|720p|1080p`: Video quality (default: best)
- `--max-videos N`: Limit number of videos from playlist
- `--audio-only`: Extract audio only (for transcription)

### Best Practices for Video Selection

**Choose videos with:**
- ✅ Clear sign language content (Ghana Sign Language)
- ✅ Good lighting and visibility
- ✅ Front-facing or clear angle view
- ✅ Minimal background distractions
- ✅ Single or few signers (easier to process)

**Avoid:**
- ❌ Multiple overlapping signers
- ❌ Poor lighting/dark videos
- ❌ Heavily edited/transition-heavy videos
- ❌ Videos with text overlays obscuring signs

## Step 2: Extract Frames

Extract quality frames from downloaded videos:
```bash
python3 scripts/extract_frames_from_videos.py --fps 1.5 --max-frames 100
```

### Options
- `--fps 1.5`: Extract 1.5 frames per second (default)
- `--max-frames N`: Maximum frames per video
- `--video filename.mp4`: Process single video

### Frame Extraction Quality Filters

Frames are automatically filtered for:
- ✅ Minimum resolution (320x240)
- ✅ Sharpness (blur detection)
- ✅ Proper brightness (not too dark/bright)
- ✅ Good contrast
- ❌ Rejects: blurry, too dark, overexposed, low contrast

## Step 3: Validate Frames

Validate extracted frames:
```bash
python3 scripts/validate_video_frames.py --organize
```

### Validation Criteria

Frames are scored on:
- **Resolution**: Prefers 1280x720 or higher (minimum 640x480)
- **Sharpness**: Blur detection (Laplacian variance > 50)
- **Brightness**: Balanced (not too dark/overexposed)
- **Contrast**: Good contrast for sign detection
- **Aspect Ratio**: Reasonable ratios (0.5-3.0)

### Frame Organization

Validated frames are organized into:
- `backend/app/data/processed/validated_frames/` - High-quality frames
- `backend/app/data/raw/rejected_frames/` - Low-quality frames

## Step 4: Integrate with Dataset

After validation, frames are ready for:
1. Manual annotation (labeling signs)
2. Linking to dictionary entries
3. Model training

## Quality Standards

### Minimum Requirements
- Resolution: ≥ 640x480
- Blur score: ≥ 50
- Brightness: 30-220
- Contrast: ≥ 20

### Recommended for Training
- Resolution: ≥ 1280x720 (HD)
- Blur score: ≥ 100
- Brightness: 50-200
- Contrast: ≥ 30
- FPS: 1-2 frames per second (sufficient for sign recognition)

## Workflow Example

```bash
# 1. Download GSL video
python3 scripts/download_youtube_sign_videos.py \
  "https://www.youtube.com/watch?v=EXAMPLE" \
  --quality 1080p

# 2. Extract frames (1.5 fps, max 200 frames)
python3 scripts/extract_frames_from_videos.py \
  --fps 1.5 \
  --max-frames 200

# 3. Validate and organize
python3 scripts/validate_video_frames.py --organize

# 4. Check results
ls backend/app/data/processed/validated_frames/ | wc -l
```

## File Structure

```
backend/app/data/
├── raw/
│   ├── youtube_videos/          # Downloaded videos
│   ├── youtube_metadata/         # Video metadata
│   ├── video_frames/             # Extracted frames (before validation)
│   └── rejected_frames/          # Low-quality frames
└── processed/
    ├── validated_frames/         # High-quality frames ready for training
    └── frame_validation_report.json
```

## Tips for High Quality

1. **Download high resolution**: Use `--quality 1080p` or `best`
2. **Extract at optimal FPS**: 1-2 fps captures signs without redundancy
3. **Validate thoroughly**: Review validation report for issues
4. **Diverse sources**: Download from multiple channels/videos
5. **Consistent lighting**: Prefer videos with good lighting
6. **Review samples**: Manually review sample frames before training

## Troubleshooting

### Video download fails
- Check internet connection
- Verify video URL is accessible
- Try different quality setting

### No frames extracted
- Check video format (some formats may not be supported)
- Verify video has playable content
- Check OpenCV installation

### All frames rejected
- Video may be too blurry/dark
- Try different video with better quality
- Adjust validation thresholds if needed

## Next Steps

After creating video dataset:
1. Annotate frames with sign labels
2. Link frames to dictionary entries
3. Combine with dictionary images for training
4. Split into train/validation/test sets

