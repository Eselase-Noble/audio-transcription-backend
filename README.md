# Audio Transcription Backend (FastAPI)

## Overview

This backend service provides APIs for **audio transcription and business document generation**.  
It accepts audio files, converts speech to text, formats the transcription into a professional business document, and generates a **downloadable PDF**.

The backend is built using **FastAPI** for high performance, scalability, and clean API architecture.

---

## Features

- 🎙️ Audio file upload (MP3, WAV, etc.)
- 🧠 Speech-to-text transcription
- 📝 Automatic business document formatting
- 📄 PDF generation
- 🚀 FastAPI-based REST API
- 🔒 CORS support for frontend integration

---

## Tech Stack

- **Python**
- **FastAPI**
- **Speech-to-Text Engine** (e.g. Whisper or similar)
- **PDF Generation Library**
- **Uvicorn** (ASGI server)

---

## Project Structure

```text
backend/
├── app/
│ ├── main.py
│ ├── routes/
│ ├── services/
│ ├── models/
│ └── utils/
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-backend-repo.git
cd backend
```

``` bash
### Set up environment variables
cp .env.example .env
### Edit .env and add your ANTHROPIC_API_KEY
```


```bash
python -m venv venv
```

``` bash
source venv/bin/activate   # macOS/Linux
```
## OR
``` bash
venv\Scripts\activate      # Windows
```
# 3. Install dependencies
``` bash
pip install -r requirements.txt
```

## Running the Server

``` bash 
uvicorn app.main:app --reload
```

```text
http://localhost:8000
```


