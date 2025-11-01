#!/usr/bin/env python3
"""
Download YouTube videos for sign language dataset.
Extracts sign language content and prepares for processing.
"""
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import re

PROJECT_ROOT = Path(__file__).parent.parent
VIDEOS_DIR = PROJECT_ROOT / "backend/app/data/raw/youtube_videos"
METADATA_DIR = PROJECT_ROOT / "backend/app/data/raw/youtube_metadata"

# Ensure directories exist
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

def check_dependencies():
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_yt_dlp():
    """Install yt-dlp if not available."""
    print("Installing yt-dlp...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'yt-dlp'], check=True)
        print("✓ yt-dlp installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install yt-dlp: {e}")
        return False

def download_video(url: str, quality: str = 'best', extract_audio: bool = False) -> dict:
    """
    Download YouTube video.
    
    Args:
        url: YouTube video URL
        quality: Video quality ('best', 'worst', '720p', '1080p', etc.)
        extract_audio: Whether to extract audio only
    
    Returns:
        Dictionary with download info
    """
    if not check_dependencies():
        if not install_yt_dlp():
            return {'error': 'yt-dlp not available and installation failed'}
    
    # Build yt-dlp command
    cmd = ['yt-dlp']
    
    # Quality options
    if extract_audio:
        cmd.extend(['-f', 'bestaudio', '--extract-audio', '--audio-format', 'mp3'])
    else:
        if quality == 'best':
            cmd.extend(['-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'])
        elif quality == 'worst':
            cmd.extend(['-f', 'worst'])
        else:
            # Specific quality
            cmd.extend(['-f', f'bestvideo[height<={quality}]+bestaudio/best[height<={quality}]'])
    
    # Output options
    output_template = str(VIDEOS_DIR / '%(title)s_%(id)s.%(ext)s')
    cmd.extend(['-o', output_template])
    
    # Get metadata
    cmd.extend(['--write-info-json', '--write-description'])
    
    # Add options to avoid 403 errors
    cmd.extend([
        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        '--extractor-args', 'youtube:player_client=android'
    ])
    
    # Add URL
    cmd.append(url)
    
    try:
        print(f"Downloading: {url}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse output to get filename
        lines = result.stdout.split('\n')
        filename = None
        for line in lines:
            if '[download]' in line and 'has already been downloaded' not in line:
                # Extract filename
                match = re.search(r'\[download\]\s+(.+)', line)
                if match:
                    filename = match.group(1).strip()
                if '%' in line:
                    continue  # Skip progress lines
                break
        
        # Get video info
        video_id = url.split('watch?v=')[-1].split('&')[0]
        info_file = METADATA_DIR / f"*_{video_id}.info.json"
        info_files = list(METADATA_DIR.glob(f"*_{video_id}.info.json"))
        
        metadata = {}
        if info_files:
            with open(info_files[0], 'r') as f:
                metadata = json.load(f)
        
        return {
            'success': True,
            'url': url,
            'video_id': video_id,
            'filename': filename,
            'metadata': metadata,
            'download_date': datetime.now().isoformat()
        }
        
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'url': url,
            'error': e.stderr or str(e)
        }


def download_playlist(playlist_url: str, quality: str = 'best', max_videos: int = None):
    """Download all videos from a YouTube playlist."""
    if not check_dependencies():
        if not install_yt_dlp():
            return []
    
    cmd = ['yt-dlp', '--flat-playlist', '--print', '%(id)s|%(title)s', playlist_url]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        video_info = []
        for line in result.stdout.strip().split('\n'):
            if '|' in line:
                vid_id, title = line.split('|', 1)
                video_info.append({'id': vid_id, 'title': title})
        
        if max_videos:
            video_info = video_info[:max_videos]
        
        print(f"Found {len(video_info)} videos in playlist")
        
        results = []
        for i, info in enumerate(video_info, 1):
            print(f"\n[{i}/{len(video_info)}] Downloading: {info['title']}")
            url = f"https://www.youtube.com/watch?v={info['id']}"
            result = download_video(url, quality)
            results.append(result)
        
        return results
        
    except subprocess.CalledProcessError as e:
        print(f"Error accessing playlist: {e}")
        return []


def save_download_log(downloads: list):
    """Save download log for reference."""
    log_file = METADATA_DIR / 'download_log.json'
    
    if log_file.exists():
        with open(log_file, 'r') as f:
            existing = json.load(f)
    else:
        existing = []
    
    existing.extend(downloads)
    
    with open(log_file, 'w') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Download log saved to {log_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download YouTube videos for sign language dataset')
    parser.add_argument('url', help='YouTube video URL or playlist URL')
    parser.add_argument('--quality', default='best', 
                        choices=['best', 'worst', '720p', '1080p', '480p'],
                        help='Video quality (default: best)')
    parser.add_argument('--playlist', action='store_true',
                        help='Download entire playlist')
    parser.add_argument('--max-videos', type=int,
                        help='Maximum videos to download from playlist')
    parser.add_argument('--audio-only', action='store_true',
                        help='Extract audio only')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YouTube Sign Language Video Downloader")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("yt-dlp not found. Installing...")
        if not install_yt_dlp():
            print("\n✗ Cannot proceed without yt-dlp")
            sys.exit(1)
    
    if args.playlist:
        results = download_playlist(args.url, args.quality, args.max_videos)
    else:
        results = [download_video(args.url, args.quality, args.audio_only)]
    
    # Filter successful downloads
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed downloads:")
        for f in failed:
            print(f"  ✗ {f.get('url', 'unknown')}: {f.get('error', 'Unknown error')}")
    
    if successful:
        save_download_log(successful)
        print(f"\n✓ Videos saved to: {VIDEOS_DIR}")
        print(f"✓ Metadata saved to: {METADATA_DIR}")
    
    print("=" * 60)

