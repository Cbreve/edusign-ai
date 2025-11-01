# 🧠 EduSign AI

**Bridging the Communication Gap with Artificial Intelligence**

---

## 🌍 Overview

EduSign AI is a bi-directional AI communication platform designed to connect deaf and hearing individuals in real time. It translates **speech and text into sign language** and also interprets **sign language gestures into speech and text**, powered by deep learning, computer vision, and 3D avatar technology.

Whether in classrooms, workplaces, or online meetings, EduSign AI ensures that everyone — regardless of hearing ability — can understand, participate, and communicate effortlessly.

---

## 💡 Problem

Deaf and hard-of-hearing individuals often face barriers in education, meetings, and daily life. Existing captioning tools only cover text, missing the full expression of tone, emotion, and gesture. There is a global need for an accessible solution that enables **two-way communication** between signers and speakers in any environment.

---

## 🚀 Solution

EduSign AI bridges this gap through an intelligent system that:

- Recognizes **speech and text**, converting them into expressive sign language via a **3D avatar**.
- Understands **sign gestures** from users and converts them into **spoken or written language**.
- Displays **real-time transcripts** alongside the avatar to ensure clarity and learning support.

It empowers full participation in classrooms, online meetings (Zoom, Google Meet, Teams), and social environments — creating an inclusive communication experience for all.

---

## 🔁 How It Works

### 1. **Speech → Sign → Transcript**
- Captures live speech using Whisper AI or Google STT.
- Converts the text into sign language animations via a 3D avatar.
- Simultaneously displays transcripts as captions for learning support.

### 2. **Sign → Speech → Transcript**
- Captures sign language gestures through MediaPipe or OpenPose.
- Interprets gestures into text using a fine-tuned WLASL-based model.
- Converts text to speech and shows a transcript confirmation.

### 3. **Multi-Modal Accessibility**
- Users can toggle transcripts, adjust signing speed, or replay translations.
- Supports multiple spoken and sign languages for global inclusivity.

---

## ✨ Key Features

- 🎙️ **Speech-to-Sign Translation** – Converts voice or text to sign language via an expressive avatar.
- ✋ **Sign-to-Speech Recognition** – Converts gestures to spoken or written words using AI vision.
- 🧍 **3D Avatar Interpreter** – Natural, emotion-aware signing with lifelike gestures.
- 💬 **Real-Time Transcripts** – Displayed alongside avatar for clarity and accessibility.
- 🌐 **Multi-Platform Support** – Integrates into Zoom, Google Meet, Teams, and web apps.
- 📶 **Offline Mode** – Works in low-connectivity areas using local AI models.
- 🧩 **Multi-Language + Multi-Sign Support** – GSL, ASL, BSL, and more.
- 🔊 **Emotion-Aware Communication** – Preserves tone and intent across translations.
- 🧠 **Adaptive Learning** – Improves translation accuracy with community feedback.
- ☁️ **Supabase Integration** – For cloud sync, user preferences, and dataset management.

---

## 🛠️ Tech Stack

**Frontend:** React.js, Next.js, Tailwind CSS  
**Backend:** FastAPI, Supabase  
**AI Components:** TensorFlow / PyTorch, MediaPipe, Whisper, gTTS  
**3D Rendering:** Three.js or Unity WebGL for avatar gestures  
**Integrations:** Zoom SDK, Google Meet API, Microsoft Teams SDK  
**Deployment:** Docker + Vercel

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- PyTorch 2.0+
- MediaPipe 0.10+

### Backend Setup

```bash
cd backend
pip install -r ../requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
npm install
npm run dev
```

---

## 🤖 Model Training

### GSL Sign Recognition Model

Our model is fine-tuned on Ghanaian Sign Language (GSL) with **95.45% validation accuracy** across **1,485 sign classes**.

**Training Pipeline:**

```bash
# Preprocess landmarks (one-time)
python scripts/preprocess_landmarks.py

# Fine-tune with WLASL pretrained model
python scripts/train_edusign_gsl.py \
    --pretrained-model backend/app/models/pretrained_wlasl.pth \
    --epochs 50 \
    --batch-size 16 \
    --fine-tune-lr 0.0001

# Test model
python scripts/inference_edusign_gsl.py \
    --model backend/app/models/edusign_gsl_finetuned.pth \
    --input <frame.jpg>
```

### Model Performance

- **Validation Accuracy**: 95.45%
- **Classes**: 1,485 GSL signs
- **Architecture**: Simplified I3D (LSTM-based)
- **Base Model**: WLASL pretrained on 2,000 ASL signs
- **Model Size**: 47MB

---

## 📊 Dataset

Our GSL dataset includes:

- **1,525 dictionary entries** with sign meanings
- **8,980 validated frames** extracted from YouTube videos
- **MediaPipe landmarks** pre-extracted and cached for fast training

### Dataset Scripts

```bash
# Extract dictionary from PDF
python scripts/extract_gsl_dictionary.py

# Download YouTube videos
python scripts/download_youtube_sign_videos.py <video_url>

# Extract and validate frames
python scripts/extract_frames_from_videos.py
python scripts/validate_video_frames.py
```

---

## 📡 API Endpoints

The FastAPI backend provides:

- **Sign Recognition**: Recognizes GSL signs from images/videos
- **Sign-to-Speech**: Converts recognized signs to speech/text
- **Speech-to-Text**: Converts speech to text
- **Text-to-Sign**: Converts text to sign language representation
- **Health Checks**: System status monitoring

See `backend/app/routes/` for endpoint definitions.

---

## 🎯 Impact

EduSign AI promotes digital inclusion by ensuring equal participation for deaf and hearing individuals. It aligns with **UN SDG 4 (Quality Education)** and **SDG 10 (Reduced Inequalities)**, making education, meetings, and communication accessible to everyone, everywhere.

---

## 🧩 Future Vision

EduSign AI aims to become a **universal AI interpreter** that supports:

- Augmented and Virtual Reality classrooms.
- Emotion-based sign interpretation.
- A global open-source sign language dataset (EduSign Corpus).
- Plug-ins for web, mobile, and smart devices.

---

## 👥 Team

**Team Cbreve** — Innovators passionate about AI, accessibility, and social impact.

---

## 📜 License

Open for academic and non-commercial use. For research or partnership inquiries, please contact the EduSign AI team.

---

## 📝 Note on Large Files

Large raw assets (PDFs, videos, images, models) are intentionally not tracked in Git (see `.gitignore`). Share via external storage or Git LFS if needed.

---

## 📚 Project Structure

```
EduSign-AI/
├── backend/          # FastAPI backend (AI & API)
├── frontend/         # Next.js frontend
├── integrations/     # SDKs for Zoom, Google Meet, Teams
├── scripts/          # Data processing and training scripts
└── docs/             # Project documentation
```

---

