# Forensic Council

**Multi-Agent Forensic Evidence Analysis System**

Welcome to the **Forensic Council** repository. This project is a production-grade, AI-powered forensic evidence analysis platform that deploys a deliberative council of specialized AI agents to analyze digital media (images, audio, video) and produce cryptographically signed, court-admissible reports.

---

## 📖 What is the Forensic Council?

### Non-Technical Overview
Imagine a team of human forensic experts (an image analyst, an audio specialist, a video expert, etc.) reviewing a piece of evidence together. They each look at it from their unique perspective, share their findings with the team, debate discrepancies, and finally, a head investigator produces a final, reliable report. 
**Forensic Council** automates this exact process using Artificial Intelligence. When you upload a file, multiple specialized AI "Agents" investigate it simultaneously. A central "Arbiter" AI then reviews their work, resolves conflicts, and generates a final, trustworthy report detailing exactly what was found.

### Technical Overview
Forensic Council is a full-stack monorepo application implementing a multi-agent system (MAS) using a structured ReAct (Reasoning + Acting) loop. 
- **The Backend (FastAPI)** orchestrates a swarm of LLM-powered agents that utilize specialized Python tools (OpenCV, librosa, metadata parsers) to analyze media. Agents communicate over an internal event bus, utilizing Redis for working memory and Qdrant for episodic vector memory. 
- **The Arbiter** consolidates agent findings, identifies conflicting conclusions, and generates a final deterministic report. Every step in the reasoning chain is cryptographically signed and stored immutably in PostgreSQL to maintain a strict chain of custody, ensuring court admissibility.
- **The Frontend (Next.js)** provides a dark-themed, interactive, glassmorphic UI where users can upload evidence, observe the agents' thought processes in real-time, step in during Human-in-the-Loop (HITL) checkpoints, and view the final generated result.

---

## 🤖 The Agents

The system relies on five distinct, highly specialized AI agents that run concurrently on the uploaded evidence:

1. **Agent 1 - Image Integrity Analyst**: A pixel-level forensic expert. It specializes in detecting image manipulation, splicing, error level analysis (ELA), and artifacts created by Generative Adversarial Networks (GANs).
2. **Agent 2 - Audio & Multimedia Expert**: Focuses on audio authenticity, waveform analysis, and detecting synthetic or artificially generated voices, as well as background noise inconsistencies.
3. **Agent 3 - Object & Weapon Specialist**: An object identification expert that looks for contextual clues, identifies specific objects (e.g., weapons, vehicles), and validates physical constraints within the media.
4. **Agent 4 - Temporal Video Analyst**: A video integrity expert analyzing motion, optical flow, frame-by-frame consistency, frame dropping, and deepfake/face-swap artifacts across time.
5. **Agent 5 - Metadata & Context Expert**: A digital footprint and provenance analyst. It extracts and validates EXIF data, GPS coordinates, timestamps, hashes, and checks for steganography or hidden structural anomalies.

---

## 📂 Project Structure

This repository is built as a complete Monorepo containing the entire stack:

```text
Agent-Council/
│
├── frontend/                   # 🖥️ Next.js 15 UI Client (React, TailwindCSS, Framer Motion)
│   ├── src/app/                # App router (Landing page, Evidence Intake, Analysis flows)
│   ├── src/components/         # Reusable UI components (Modals, Panels, Agent visualizers)
│   ├── src/hooks/              # Custom React hooks (Store, Simulation, etc.)
│   └── public/                 # Static assets
│
├── backend/                    # ⚙️ FastAPI Python Backend (LLM Orchestration)
│   ├── agents/                 # The 5 specialized AI agents + base classes
│   ├── tools/                  # Python functions (Image, Audio, Video, Metadata tools) utilized by agents
│   ├── core/                   # Core system logic (Inter-Agent bus, ReAct loop, Logging, config)
│   ├── infra/                  # Infrastructure wrappers (Redis, Qdrant, PostgreSQL, Storage)
│   ├── orchestration/          # Main execution pipelines and session management
│   ├── reports/                # Final cryptographic report formatting and rendering
│   └── scripts/                # Utility scripts (e.g., Database initialization)
│
├── docs/                       # 📚 Architectural decisions, system flow, and roadmaps
├── docker-compose.yml          # 🐳 Compose file to spin up the entire infrastructure locally
└── README.md                   # You are here!
```

---

## 🚀 Quickstart

The easiest way to boot the entire stack is via Docker Compose.

1. **Environment Setup**
   Navigate to the `/backend` specific directory and set up your environment variables based on `.env.example`:
   ```bash
   cp backend/.env.example backend/.env
   # Make sure to edit backend/.env and add your OPENAI_API_KEY
   ```

2. **Boot the Application via Docker**
   From the **root** of this repository, run:
   ```bash
   docker compose up --build
   ```

3. **Access the Interface**
   - **Frontend UI:** Navigate to [http://localhost:3000](http://localhost:3000)
   - **Backend API Docs:** Navigate to [http://localhost:8000/docs](http://localhost:8000/docs)

Once booted, navigate to the frontend UI, drop an image or video file into the system, and watch the agents collaborate in real-time to analyze your evidence!
