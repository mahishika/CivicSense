# CivicSense

AI-powered civic issue reporting and verification platform. Citizens report problems (potholes, water leaks, garbage, streetlights, etc.) via text, voice, or photo in any Indian language. AI agents classify, route, and prioritize each report; authorities resolve issues and AI verifies the fix using before/after photos.

## Features

- Multilingual reporting (9 Indian languages) with voice-to-text input
- Photo evidence upload for reports and resolutions
- Live GPS or manual location selection, automatically matched to the right jurisdiction
- AI-powered classification, sentiment analysis, and urgency scoring (Gemini 2.5 Flash)
- Admin dashboard scoped to each authority's jurisdiction, sorted by urgency
- AI-verified resolution using before/after photo comparison
- Citizen trust scoring system
- Anonymous reporting option

## Tech Stack

- **Backend:** FastAPI (Python)
- **AI:** Google Gemini 2.5 Flash
- **Frontend:** Plain HTML + Tailwind CSS (no build step)
- **Deployment:** Google Cloud Run

## Project Structure

```
CivicSense/
├── main.py              # FastAPI backend with AI agent pipeline
├── frontend.html         # Single-page frontend (citizen + admin views)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container build for Cloud Run
├── .env.example          # Environment variable template
└── .gitignore
```

## Local Setup

1. Clone the repo and enter the folder:
   ```bash
   git clone https://github.com/mahishika/CivicSense.git
   cd CivicSense
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Copy the environment template and add your Gemini API key:
   ```bash
   cp .env.example .env
   ```
   Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey) and paste it into `.env` as `GEMINI_API_KEY=...`.

4. Run the backend:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

5. Serve the frontend (in a separate terminal):
   ```bash
   python -m http.server 5500
   ```
   Then open `http://localhost:5500/frontend.html` in your browser.

## Deployment (Google Cloud Run)

```bash
gcloud auth login
gcloud config set project <your-project-id>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

gcloud run deploy civicsense-backend \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your_key_here
```

After deployment, update `API_URL` in `frontend.html` to point to the Cloud Run service URL printed at the end of the deploy command.

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Required. Gemini API key for the AI agent pipeline. |

## License

This project was built for a hackathon submission and is provided as-is.
