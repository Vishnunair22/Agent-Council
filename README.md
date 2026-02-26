# Forensic Council

Welcome to Forensic Council, a multi-agent AI system designed to analyze digital evidence (images, audio, video) and produce cryptographically signed chain-of-custody reports.

This repository is built as a Monorepo containing the full stack:
- `/frontend`: The Next.js 15 UI client for uploading evidence and viewing the ReAct AI thought streams.
- `/forensic_council`: The FastAPI backend running the AI orchestration, connecting to PostgreSQL, Redis, and Qdrant.
- `/docs`: Architectural and roadmap documentation.

## Quickstart

The easiest way to boot the entire stack is via Docker Compose. Ensure you have Docker and Docker Compose installed on your machine.

1. **Environment Variables**
   Open the `/forensic_council/.env` file and ensure you have populated the required keys:
   ```env
   OPENAI_API_KEY=your_key_here
   SIGNING_KEY=your_secure_development_key
   ```

2. **Boot the Application**
   Run the following command from the root of this repository:
   ```bash
   docker compose up --build
   ```

3. **Access the App**
   - **Frontend UI:** [http://localhost:3000](http://localhost:3000)
   - **Backend API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

Once booted, you can drag and drop digital media into the UI to witness the Council Arbiter, Video, Image, Audio, and Metadata agents collaborate in real-time.
