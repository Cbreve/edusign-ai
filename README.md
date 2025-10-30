## EduSign AI: Bridging the Communication Gap for Inclusive Learning

### Overview
EduSign AI is a smart communication platform designed to empower deaf and hard-of-hearing individuals to learn, interact, and communicate seamlessly. It converts speech and text into sign language in real time using artificial intelligence and 3D avatar technology. Whether in classrooms, workplaces, or virtual meetings, EduSign AI ensures that communication is clear, natural, and inclusive.

### Problem Statement
People with hearing impairments face daily communication barriers in education, work, and social life. In schools, students often struggle to follow lessons without interpreters. In workplaces or virtual meetings, understanding spoken communication becomes even harder. Current tools like captions or subtitles only provide partial meaning and fail to express tone or emotion. There is a need for a solution that accurately translates both speech and emotion into sign language instantly, anywhere.

### Solution
EduSign AI bridges communication gaps through an AI-powered system that recognizes speech, processes it into text, and converts it into sign language performed by an expressive 3D avatar. The avatar reproduces natural gestures and facial expressions to preserve meaning and context. The system integrates smoothly into classrooms, workplaces, and online meeting platforms, promoting inclusion and accessibility.

### Supported Environments
- Physical Classrooms: Converts teachers’ speech into on-screen sign language for students to follow lessons easily.
- Online Meetings: Integrates with Zoom, Google Meet, and Microsoft Teams so deaf participants can understand discussions in real time.
- Daily Communication: Facilitates one-on-one interactions in schools, offices, and public spaces.

### Key Features
- Speech-to-Sign Translation: Converts spoken or written text into sign language instantly using AI-driven interpretation.
- 3D Avatar Interpreter: Displays lifelike gestures and emotions to make conversations engaging and meaningful.
- Multi-Platform Integration: Works with major meeting tools like Zoom, Google Meet, and Microsoft Teams for accessible online experiences.
- Offline Access: Provides local translation without internet connectivity for rural or low-bandwidth areas.
- Multi-Language and Multi-Sign Support: Includes several spoken and sign languages such as ASL, BSL, and GSL for broader inclusivity.
- Emotion-Aware Communication: Interprets tone and sentiment to ensure emotional context is preserved.
- Educational Mode: Allows teachers to upload notes or videos for automatic translation into sign language.
- Customizable Interface: Lets users adjust signing speed, avatar style, and text visibility based on preference.
- Community Collaboration: Enables users, interpreters, and educators to share signs, improve translations, and contribute to a growing open-source sign database.
- SDK & API Integration: Offers tools for developers to embed EduSign AI features into learning platforms and digital classrooms.
- Augmented and Virtual Reality Support: Creates immersive learning environments for deeper engagement.

### Technology Stack
- Frontend: React.js, Next.js, Tailwind CSS
- Backend: FastAPI, Supabase
- AI Engine: Speech recognition, natural language processing, and vision-based sign synthesis powered by TensorFlow and MediaPipe
- Integration: APIs for meeting and learning platforms; Firebase and Vercel for scalable deployment

### Impact
EduSign AI creates opportunities for inclusive education and work by giving deaf and hard-of-hearing individuals equal access to communication. It allows learners to participate fully in classrooms, meetings, and discussions without constant interpretation. By combining AI, accessibility, and cross-platform integration, EduSign AI supports global inclusion and aligns with Sustainable Development Goal 4 (Quality Education).

### Target Users
- Deaf and hard-of-hearing individuals
- Students and educators in inclusive classrooms
- Organizations hosting meetings and training sessions
- Event organizers and NGOs promoting accessibility
- Institutions and governments focused on inclusive digital education

### Vision
EduSign AI envisions a world where communication barriers no longer exist. Every person deserves to be heard, understood, and included—whether in learning, work, or daily life. By combining AI innovation with empathy and cultural understanding, EduSign AI transforms accessibility into equality for all.

### Closing Statement
EduSign AI is more than a translation tool; it is a movement toward inclusive communication worldwide. Through cutting-edge AI, collaboration, and community-driven design, it connects people, bridges silence, and ensures that every voice—spoken or signed—can be understood.

## Developer guide

### Processed outputs
- `backend/app/data/processed/gsl_dictionary.json`
- `backend/app/data/processed/gsl_dictionary.csv`

### Raw data (not stored in Git)
Large raw assets (PDFs, extracted images) are intentionally not tracked in Git to keep the repository lightweight and within platform limits.

- Place the raw PDF at `backend/app/data/raw/Harmonized_GSL_Dictionary_v3_2023.pdf` (or `Harmonized_GSL_Dictionary.pdf`).
- The `.gitignore` prevents committing large PDFs and `sign_images/`.
- If the raw file is needed by multiple contributors, share it via external storage (e.g., Drive/S3) and download locally before running the extractor.

### Running the extractor
```
python backend/app/utils/extract_gsl_dictionary.py
```
This generates the processed JSON/CSV and prints a brief summary.

### Data quality checks
- The extractor applies noise filters, column-aware parsing, optional OCR fallback with confidence gating, and adds metadata per entry.
- A small gold set lives at `backend/app/data/gold/gsl_gold.json`. The CI workflow validates extractor output against this set and fails when any item is missing.

### CI
GitHub Actions workflow (`.github/workflows/quality.yml`) installs dependencies, runs the extractor, and validates against the gold set on every push/PR to `main`.
