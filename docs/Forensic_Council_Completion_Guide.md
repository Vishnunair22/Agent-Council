# Forensic Council — Completion Guide
## From Current State to Final Running Application
### Version 1.0 · Build, Wire, Test, Run

---

## How to Use This Document

This guide picks up exactly where your current project state ends. It is structured as a sequence of **Kilo Code task prompts** (for backend) and **direct instructions** (for frontend fixes and integration), each with a precise test gate before proceeding.

**Read the Current State Assessment first.** It documents exactly what exists, what is broken, and what is missing before a single line of code is written.

---

## Current State Assessment

### What Exists

| Layer | Status | Detail |
|---|---|---|
| **Architecture Document** | ✅ Complete | `Forensic_Council_Architecture.md` — all 11 principles defined |
| **Implementation Roadmap** | ✅ Complete | `Forensic_Council_Implementation_Roadmap.md` — 12 Kilo Code stages |
| **Docker Infrastructure** | ✅ Ready | `docker-compose.yml` — Redis, Qdrant, PostgreSQL defined |
| **Python Project Config** | ✅ Ready | `pyproject.toml` — all dependencies declared |
| **Environment Template** | ✅ Ready | `.env.example` — all vars templated |
| **Frontend Mock** | ✅ Built (broken build) | Next.js 16, React 19, Tailwind v4, Framer Motion |
| **Backend Python Code** | ❌ Not written | All 12 stages from roadmap still to be executed |
| **FastAPI Layer** | ❌ Missing entirely | No API bridge between frontend and backend exists |
| **Frontend → API wiring** | ❌ Simulated only | `useSimulation.ts` uses `setTimeout`, `useForensicData.ts` uses `localStorage` |
| **Regression Test Suite** | ❌ Not written | No frontend tests, no integration tests |

### Critical Issues to Fix Before Anything Else

**Issue 1 — Frontend Build Failure (BLOCKER)**
All client-side pages and hooks are missing the `"use client"` directive required by Next.js App Router. The build currently fails with 8 errors across:
- `src/app/evidence/page.tsx`
- `src/app/result/page.tsx`
- `src/hooks/useForensicData.ts`
- `src/hooks/useSimulation.ts`

**Issue 2 — Redis Port Mismatch (SILENT BUG)**
`docker-compose.yml` maps Redis as `6380:6379` (external port 6380) but `.env.example` sets `REDIS_PORT=6379`. Your Python backend will fail to connect to Redis unless both are aligned.

**Issue 3 — No API Layer**
The frontend dev guide explicitly notes that `useSimulation.ts` and `useForensicData.ts` have `Backend TODO` comments. There is no FastAPI (or equivalent) server and no Next.js API routes. Without this, the frontend cannot communicate with the Python pipeline regardless of how complete the backend becomes.

---

## Project Folder Structure — Target State

Before starting, understand the final target structure:

```
backend/                    ← Python backend root
├── core/
├── infra/
├── agents/
├── tools/
├── orchestration/
├── reports/
├── api/                             ← NEW: FastAPI server (Stage 13)
│   ├── __init__.py
│   ├── main.py
│   ├── routes/
│   │   ├── investigation.py
│   │   ├── sessions.py
│   │   └── hitl.py
│   └── schemas.py
├── tests/
├── scripts/
├── docker-compose.yml
├── pyproject.toml
└── .env

agent-council-front-end/             ← Next.js frontend root
├── src/
│   ├── app/
│   │   ├── evidence/page.tsx        ← Fix: add "use client"
│   │   ├── result/page.tsx          ← Fix: add "use client"
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/
│   ├── hooks/
│   │   ├── useForensicData.ts       ← Fix: add "use client" + real API calls
│   │   └── useSimulation.ts         ← Fix: add "use client" + SSE/WebSocket
│   ├── lib/
│   │   ├── api.ts                   ← NEW: API client module
│   │   ├── constants.ts
│   │   ├── schemas.ts
│   │   └── utils.ts
│   └── types/
│       └── index.ts                 ← Update: align with backend response shape
├── package.json
└── .env.local                       ← NEW: NEXT_PUBLIC_API_URL
```

---

## PRE-STEP — Fix Critical Issues Before Any Coding

### PRE-1: Fix Redis Port Mismatch

Open your `.env` file and set:
```
REDIS_PORT=6380
```

Or alternatively, change `docker-compose.yml` Redis port mapping from `6380:6379` to `6379:6379` and keep `REDIS_PORT=6379`. Pick one — this guide assumes you change the `.env` to match docker-compose (`REDIS_PORT=6380`).

---

### PRE-2: Fix Frontend Build Errors

**Paste this prompt into Kilo Code (frontend project):**

```
Fix all "use client" directive errors in the Next.js frontend.

The following files use React hooks (useState, useEffect, useRef, useCallback)
and are being treated as Server Components by Next.js App Router, causing a
build failure. Add "use client"; as the very first line of each file:

1. src/app/evidence/page.tsx — add "use client"; as line 1
2. src/app/result/page.tsx — add "use client"; as line 1
3. src/hooks/useForensicData.ts — add "use client"; as line 1
4. src/hooks/useSimulation.ts — add "use client"; as line 1

Do not change any other code. Only add the directive.
```

**Test Gate:**
```bash
cd agent-council-front-end
npm run build
```
Expected: Build completes with 0 errors. You will see route compilation output with no red error blocks.

---

## BACKEND BUILD — Execute Stages 0–12

Execute every stage from `Forensic_Council_Implementation_Roadmap.md` in exact order. Each stage prompt is already written in that document. Paste each into Kilo Code and run the test gate before proceeding.

This section provides the stage dependency checklist and gate commands only. Do not skip stages or run them out of order — each stage is a dependency of the next.

```
Stage 0  → Infrastructure skeleton + Docker + DB schema
Stage 1  → Cryptographic signing + Chain-of-custody logger
Stage 2  → Evidence ingestion + Immutable versioning
Stage 3  → Working memory (Redis) + Episodic memory (Qdrant)
Stage 4  → ReAct loop engine + HITL checkpoint system
Stage 5  → ForensicAgent base class + Self-reflection
Stage 6  → Real tools: Image tools (Agent 1) + Metadata tools (Agent 5)
Stage 7  → Real tools: Audio tools (Agent 2) + Video tools (Agent 4)
Stage 8  → Inter-agent communication bus
Stage 9  → Confidence calibration layer
Stage 10 → Adversarial robustness checks
Stage 11 → Council Arbiter + Report generator
Stage 12 → End-to-end integration test
```

**After every stage, commit to git:**
```bash
git add -A
git commit -m "Stage N complete — X tests passing"
```

**After Stage 12, verify the full backend suite:**
```bash
cd backend
docker compose up -d
uv run pytest tests/ -v --cov=. --cov-report=term-missing
```
Expected: All tests pass, coverage ≥ 75%, report renders in terminal.

---

## Stage 13 — FastAPI Backend Server

This stage is **not in the original roadmap** but is required to connect the frontend to the backend pipeline.

### 13.1 — Kilo Code Prompt

```
Build a FastAPI HTTP + WebSocket server for the forensic_council project.
This server is the bridge between the Next.js frontend and the Python pipeline.

Add to pyproject.toml dependencies:
- fastapi>=0.115
- uvicorn[standard]>=0.30
- python-multipart>=0.0.9   (required for file upload)
- websockets>=13.0

Create api/__init__.py (empty)

Create api/schemas.py

InvestigationRequest: Pydantic model
  {case_id: str, investigator_id: str}

InvestigationResponse: Pydantic model
  {session_id: str, case_id: str, status: str, message: str}

AgentFindingDTO: Pydantic model (serializable subset of AgentFinding for frontend)
  {finding_id: str, agent_id: str, agent_name: str,
   finding_type: str, status: str,
   confidence_raw: float, calibrated: bool,
   calibrated_probability: float | None,
   court_statement: str | None,
   robustness_caveat: bool,
   robustness_caveat_detail: str | None,
   reasoning_summary: str}

ReportDTO: Pydantic model (serializable subset of ForensicReport for frontend)
  {report_id: str, session_id: str, case_id: str,
   executive_summary: str,
   per_agent_findings: dict[str, list[AgentFindingDTO]],
   cross_modal_confirmed: list[AgentFindingDTO],
   contested_findings: list[dict],
   tribunal_resolved: list[dict],
   incomplete_findings: list[AgentFindingDTO],
   uncertainty_statement: str,
   cryptographic_signature: str,
   report_hash: str,
   signed_utc: str}

HITLDecisionRequest: Pydantic model
  {session_id: str, checkpoint_id: str, agent_id: str,
   decision: Literal["APPROVE","REDIRECT","OVERRIDE","TERMINATE","TRIBUNAL"],
   note: str | None, override_finding: dict | None}

BriefUpdate: Pydantic model (WebSocket message)
  {type: Literal["AGENT_UPDATE","HITL_CHECKPOINT","AGENT_COMPLETE","PIPELINE_COMPLETE","ERROR"],
   session_id: str, agent_id: str | None, agent_name: str | None,
   message: str, data: dict | None}

Create api/routes/investigation.py

POST /api/v1/investigate
  - Accepts: multipart/form-data with file (evidence) + case_id + investigator_id
  - Validates file size ≤ 50MB
  - Validates MIME type: image/jpeg, image/png, video/mp4, audio/wav, audio/mpeg
  - Ingests evidence via EvidenceStore
  - Starts ForensicCouncilPipeline.run_investigation() as background task
    (asyncio.create_task — non-blocking, returns immediately)
  - Returns: InvestigationResponse with session_id

GET /api/v1/sessions/{session_id}/report
  - Returns: ReportDTO if investigation is complete
  - Returns 202 Accepted with {"status": "in_progress"} if still running
  - Returns 404 if session_id not found

GET /api/v1/sessions/{session_id}/brief/{agent_id}
  - Returns current investigator brief text for agent
  - Returns 404 if not found

GET /api/v1/sessions/{session_id}/checkpoints
  - Returns list of pending HITL checkpoints
  - Returns [] if none pending

Create api/routes/hitl.py

POST /api/v1/hitl/decision
  - Accepts: HITLDecisionRequest
  - Routes decision to correct agent via pipeline.handle_hitl_decision()
  - Returns 200 on success

Create api/routes/sessions.py

GET /api/v1/sessions
  - Returns list of all active sessions (session_id, case_id, status, started_at)

DELETE /api/v1/sessions/{session_id}
  - Terminates a session if running

Create api/main.py

FastAPI app with:
  - CORS middleware: allow origins ["http://localhost:3000"] for development
  - Include routers: investigation, hitl, sessions
  - Lifespan context manager: initialize DB, Redis, Qdrant on startup; close on shutdown

WebSocket endpoint: WS /api/v1/sessions/{session_id}/live
  - Client connects after starting investigation
  - Server sends BriefUpdate messages as agents produce findings
  - Messages sent for:
    * Each agent THOUGHT/ACTION/OBSERVATION cycle summary (every 3 iterations, not every step — prevents flooding)
    * HITL checkpoint triggers (type=HITL_CHECKPOINT, includes checkpoint_id and agent brief)
    * Agent loop completion (type=AGENT_COMPLETE, includes finding count)
    * Pipeline completion (type=PIPELINE_COMPLETE, report_id embedded)
    * Any error (type=ERROR)
  - Session state is managed via SessionManager
  - Connection closed gracefully when pipeline completes

Add script: scripts/run_api.py
  import uvicorn
  uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

Tests in tests/test_api/test_investigation.py:
Use httpx.AsyncClient with the FastAPI test client.
- test_post_investigate_valid_image_returns_session_id
- test_post_investigate_invalid_mime_type_returns_422
- test_post_investigate_file_too_large_returns_413
- test_get_report_in_progress_returns_202
- test_get_report_complete_returns_report_dto
- test_get_brief_returns_text
- test_get_checkpoints_returns_list
- test_post_hitl_decision_approve_returns_200
- test_websocket_connects_and_receives_updates (mock pipeline, assert BriefUpdate messages received)
```

### 13.2 — Test Gate

```bash
cd backend

# Start the API server
uv run python scripts/run_api.py &

# Verify server is running
curl http://localhost:8000/docs

# Run API tests
uv run pytest tests/test_api/ -v

# Test file upload manually
curl -X POST http://localhost:8000/api/v1/investigate \
  -F "file=@tests/fixtures/test_spliced_image.jpg" \
  -F "case_id=MANUAL-TEST-001" \
  -F "investigator_id=DEVELOPER"

# Kill background server
kill %1
```

Expected: `/docs` opens Swagger UI. All 9 API tests pass. File upload returns JSON with `session_id`.

### 13.3 — Stage Completion Report Prompt

```
Stage 13 complete. Report:

BUILT: all files created
TESTS: all test names and status
ENDPOINTS: List all registered routes with method and path
WEBSOCKET: Confirm WebSocket test received at least one BriefUpdate message
CORS: Confirm localhost:3000 is in allowed origins
FULL SUITE: X tests, 0 failing
WHAT REMAINS: Stage 14 — Frontend API integration
```

---

## Stage 14 — Frontend API Integration

This stage replaces all mock/simulation code in the frontend with real API calls to the Stage 13 FastAPI server.

### 14.1 — Kilo Code Prompt

```
Wire the Next.js frontend to the real FastAPI backend.

Create src/lib/api.ts

This is the API client module. All backend communication goes through this file.

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
const WS_BASE  = API_BASE.replace("http", "ws")

export async function startInvestigation(
  file: File, caseId: string, investigatorId: string
): Promise<{ session_id: string }>
  POST multipart/form-data to /api/v1/investigate
  Returns session_id from response JSON

export async function getReport(
  sessionId: string
): Promise<{ status: "complete" | "in_progress"; report?: ReportDTO }>
  GET /api/v1/sessions/{sessionId}/report
  If 202: return {status: "in_progress"}
  If 200: return {status: "complete", report: data}
  If error: throw

export async function getBrief(
  sessionId: string, agentId: string
): Promise<string>
  GET /api/v1/sessions/{sessionId}/brief/{agentId}
  Returns brief text

export async function getCheckpoints(
  sessionId: string
): Promise<HITLCheckpoint[]>
  GET /api/v1/sessions/{sessionId}/checkpoints
  Returns list of pending checkpoints

export async function submitHITLDecision(
  decision: HITLDecisionRequest
): Promise<void>
  POST /api/v1/hitl/decision

export function createLiveSocket(
  sessionId: string,
  onMessage: (update: BriefUpdate) => void,
  onClose: () => void
): WebSocket
  Opens WS connection to /api/v1/sessions/{sessionId}/live
  Parses incoming JSON as BriefUpdate
  Calls onMessage for each message
  Calls onClose on connection close

Update src/types/index.ts

Add types that match backend DTOs exactly:
- AgentFindingDTO (matches api/schemas.py AgentFindingDTO)
- ReportDTO (matches api/schemas.py ReportDTO)
- BriefUpdate (matches api/schemas.py BriefUpdate)
- HITLCheckpoint (pending checkpoint shape)
- HITLDecisionRequest (matches api/schemas.py HITLDecisionRequest)

Keep existing Report and AgentResult types but mark as @deprecated
(they were the mock simulation types)

Update src/hooks/useSimulation.ts

Replace the setTimeout simulation with real WebSocket connection:

const useSimulation = (sessionId: string | null) => {
  - When sessionId is provided, open WebSocket via createLiveSocket()
  - Map BriefUpdate messages to existing agent state shape so UI renders correctly:
    * type=AGENT_UPDATE → update that agent's "thinking" text and status to "analyzing"
    * type=HITL_CHECKPOINT → set a hitlCheckpoint state with the checkpoint data
    * type=AGENT_COMPLETE → set that agent's status to "complete"
    * type=PIPELINE_COMPLETE → set overall status to "complete", store report_id
  - Keep the same state shape (agents array with status/thinking fields) so
    existing UI components work without changes
  - Close WebSocket on unmount or when pipeline_complete received
}

Update src/hooks/useForensicData.ts

Replace localStorage simulation with real API calls:

const useForensicData = () => {
  - startAnalysis(file, caseId, investigatorId):
    * Call startInvestigation() from api.ts
    * Store returned session_id in component state (not localStorage)
    * Return session_id

  - pollForReport(sessionId):
    * Poll getReport() every 5 seconds
    * When status="complete": save report to state
    * Stop polling

  - Keep saveCurrentReport and addToHistory but store in React state (not localStorage)
    so the result page can read them during the same session
  
  - getCurrentReport(): returns current report from state
  - getHistory(): returns history from state

  Note: Cross-session persistence (localStorage) is a future enhancement.
  For now, data lives in React state for the duration of the session.
}

Update src/app/evidence/page.tsx

Replace the existing simulation trigger logic:
- On file upload submit:
  1. Call useForensicData().startAnalysis(file, caseId, investigatorId)
  2. Get session_id back
  3. Pass session_id to useSimulation(session_id) to open WebSocket
  4. Navigate to /result when pipeline_complete WebSocket message received

Update src/app/result/page.tsx

Replace localStorage read with useForensicData().getCurrentReport()
Map ReportDTO fields to the existing display components:
- executive_summary → existing summary display
- per_agent_findings → existing agent results list
  Map each AgentFindingDTO to AgentResult shape
- court_statement → new display section (add below existing findings)
- uncertainty_statement → new display section
- cryptographic_signature + report_hash → new "Verification" section at bottom

Add HITL checkpoint UI:
- If useSimulation returns a pending hitlCheckpoint:
  Show a modal overlay with the checkpoint brief text
  and buttons: Approve / Redirect / Terminate / Escalate to Tribunal
  On button click: call submitHITLDecision() from api.ts

Add export buttons (already in UI but need wiring):
- "Export JSON": JSON.stringify(currentReport) and trigger download
- "Export PDF": POST report to /api/v1/reports/{report_id}/pdf (implement endpoint in Stage 13 extension if needed, or show "coming soon")

Create .env.local in frontend root:
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 14.2 — Test Gate

```bash
cd agent-council-front-end

# Verify build still passes
npm run build

# Start frontend dev server
npm run dev &

# Start backend in a separate terminal
cd ../backend
docker compose up -d
uv run python scripts/run_api.py &

# Manual integration smoke test
# 1. Open http://localhost:3000
# 2. Upload a test image
# 3. Verify agent cards animate and show real "thinking" text from WebSocket
# 4. Verify result page shows real report data (not mock)
# 5. Verify HITL modal appears if a checkpoint is triggered

# Kill dev servers
kill %1 %2
```

### 14.3 — Stage Completion Report Prompt

```
Stage 14 complete. Report:

BUILT: all files created or modified
API CLIENT: Confirm api.ts exports all 6 functions
WEBSOCKET: Confirm useSimulation now opens a WebSocket (not setTimeout)
DATA FLOW: Confirm useForensicData no longer reads/writes localStorage
BUILD: npm run build passes with 0 errors
HITL UI: Confirm HITL checkpoint modal renders correctly
REPORT DISPLAY: List all new sections added to result page (court_statement, uncertainty_statement, verification block)
WHAT REMAINS: Regression testing and quality checks (Stage 15)
```

---

## Stage 15 — Regression Testing & Quality Checks

This is the quality assurance stage. It adds the full regression test suite, enforces coverage thresholds, and runs linting and type checks across both projects.

### 15.1 — Backend Regression Test Suite

**Paste into Kilo Code:**

```
Add a regression test suite to the forensic_council project.
This suite runs after every code change to prevent regressions.

Create tests/test_regression/

tests/test_regression/test_core_regression.py

These tests verify that core components work correctly end-to-end
and have not regressed from a previous working state.

- test_signing_chain_is_tamper_evident
  Create a signed chain of 10 entries, corrupt entry 5, verify
  verify_chain() returns broken_at = entry 5's ID.
  Assert entries 1-4 are valid and entries 6-10 are also valid
  (corruption is isolated, not contagious).

- test_working_memory_survives_hitl_pause_resume_cycle
  Run a full serialize → clear → restore cycle.
  Assert all 8 task statuses, current_iteration, and hitl_state
  are byte-for-byte identical after restore.

- test_evidence_version_tree_integrity_under_three_levels
  Ingest a file, create 2 derivative levels (3 nodes total).
  Verify each node's parent_id links correctly.
  Modify the leaf node's stored file.
  Assert verify_artifact_integrity returns False only for leaf node.
  Assert root and middle nodes still pass integrity check.

- test_reagent_loop_iteration_ceiling_is_enforced
  Configure a loop with ceiling=5.
  Run a mock agent that never produces a finding.
  Assert loop terminates at exactly iteration 5.
  Assert final result has completed=False and total_iterations=5.
  Assert last chain entry is type HITL_CHECKPOINT (escalated).

- test_calibration_version_immutability
  Create version 1.0 of a calibration model.
  Fit version 2.0.
  Assert version 1.0 is still loadable and its params are unchanged.
  Assert a finding produced under version 1.0 still references version 1.0
  when retrieved after version 2.0 exists.

- test_inter_agent_bus_blocks_all_unpermitted_paths
  Test every agent-to-agent pair that is NOT in PERMITTED_CALL_PATHS.
  Assert PermittedCallViolationError raised for each.
  Expected blocked paths: Agent1→Agent2, Agent1→Agent3, Agent1→Agent4,
  Agent1→Agent5, Agent2→Agent3, Agent2→Agent5, Agent3→Agent2,
  Agent3→Agent4, Agent3→Agent5, Agent4→Agent3, Agent4→Agent5,
  Agent5→(all others)

- test_arbiter_never_silently_resolves_contested_finding
  Construct two AgentFinding objects with same evidence region but
  contradictory conclusions (CONFIRMED vs INCONCLUSIVE).
  Run cross_agent_comparison().
  Assert output contains exactly one FindingComparison with
  verdict=CONTRADICTION.
  Assert neither finding has been mutated or merged.

- test_graceful_degradation_produces_incomplete_finding_not_exception
  Register a tool as unavailable in ToolRegistry.
  Run an agent loop where that tool is the only tool for a mandatory task.
  Assert loop completes without raising an exception.
  Assert findings list contains exactly one finding with
  status=INCOMPLETE.
  Assert working memory shows that task as BLOCKED.

tests/test_regression/test_api_regression.py

- test_file_upload_produces_signed_report_end_to_end
  POST a real test image to /api/v1/investigate.
  Poll /api/v1/sessions/{session_id}/report until complete (max 120s timeout).
  Assert report.cryptographic_signature is not empty string.
  Assert report.report_hash matches SHA-256 of report content fields.
  Assert report.executive_summary is not empty string.

- test_hitl_checkpoint_pause_resume_via_api
  POST a test image.
  Wait for a HITL_CHECKPOINT WebSocket message.
  POST APPROVE decision to /api/v1/hitl/decision.
  Assert investigation continues (next AGENT_UPDATE or PIPELINE_COMPLETE received).

- test_concurrent_sessions_do_not_cross_contaminate
  Start two simultaneous investigations with different files.
  Assert each session_id returns a different report.
  Assert chain-of-custody logs are fully separate per session.

Add to pyproject.toml under [tool.pytest.ini_options]:
  markers = [
    "regression: marks tests as regression tests (deselect with -m 'not regression')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
  ]

Mark all existing tests appropriately:
  tests/test_core/     → @pytest.mark.unit
  tests/test_infra/    → @pytest.mark.unit
  tests/test_tools/    → @pytest.mark.unit
  tests/test_agents/   → @pytest.mark.unit
  tests/test_api/      → @pytest.mark.integration
  tests/test_integration/ → @pytest.mark.integration
  tests/test_regression/  → @pytest.mark.regression
```

**Test Gate:**
```bash
cd backend

# Run unit tests only (fast — no inference)
uv run pytest tests/ -m "unit" -v --tb=short

# Run integration tests (requires running infrastructure)
docker compose up -d
uv run pytest tests/ -m "integration" -v --tb=short

# Run regression suite (slowest — real pipeline runs)
uv run pytest tests/ -m "regression" -v --tb=short -s

# Full suite with coverage
uv run pytest tests/ -v --cov=. --cov-report=term-missing --cov-fail-under=75

# Run regression suite in isolation to verify no regressions from prior stages
uv run pytest tests/test_regression/ -v
```

Expected: All regression tests pass. Coverage ≥ 75%. Zero test failures across all markers.

---

### 15.2 — Frontend Quality Checks

**Paste into Kilo Code (frontend project):**

```
Add frontend quality checks and basic test coverage.

Install test dependencies:
npm install --save-dev jest jest-environment-jsdom @testing-library/react
    @testing-library/jest-dom @testing-library/user-event ts-jest

Create jest.config.ts:
  testEnvironment: jsdom
  transform: ts-jest
  moduleNameMapper: { "^@/(.*)$": "<rootDir>/src/$1" }
  setupFilesAfterFramework: ["@testing-library/jest-dom"]

Create src/__tests__/lib/api.test.ts

Mock global fetch. Test:
- test_startInvestigation_sends_correct_multipart_form_data
  Mock fetch to return { session_id: "test-123" }.
  Call startInvestigation(mockFile, "CASE-1", "INVESTIGATOR-1").
  Assert fetch called with POST, correct URL.
  Assert FormData contained file, case_id, investigator_id fields.
  Assert returned session_id equals "test-123".

- test_getReport_returns_in_progress_on_202
  Mock fetch to return status 202.
  Assert getReport("any-id") returns { status: "in_progress" }.

- test_getReport_returns_complete_with_report_on_200
  Mock fetch to return status 200 with mock ReportDTO JSON.
  Assert getReport("any-id") returns { status: "complete", report: mockReport }.

- test_startInvestigation_throws_on_server_error
  Mock fetch to return status 500.
  Assert startInvestigation() throws an error.

Create src/__tests__/hooks/useForensicData.test.ts

- test_startAnalysis_calls_api_and_returns_session_id
  Render hook, mock startInvestigation to return { session_id: "abc" }.
  Call startAnalysis(mockFile, "CASE", "INV").
  Assert returned value is "abc".

- test_getCurrentReport_returns_null_before_complete
  Render hook without calling pollForReport.
  Assert getCurrentReport() returns null.

Create src/__tests__/types/schema.test.ts

- test_ReportDTO_zod_schema_validates_complete_report
  Create a complete mock ReportDTO object.
  Run it through the Zod schema (if schemas.ts exports one for ReportDTO).
  Assert validation passes.

- test_ReportDTO_zod_schema_rejects_missing_required_fields
  Remove executive_summary from the mock DTO.
  Assert validation fails with correct field error.

Add to package.json scripts:
  "test": "jest"
  "test:watch": "jest --watch"
  "test:coverage": "jest --coverage"
  "type-check": "tsc --noEmit"
  "lint": "eslint src/ --ext .ts,.tsx"
```

**Test Gate:**
```bash
cd agent-council-front-end

# TypeScript type check
npm run type-check

# Lint
npm run lint

# Unit tests
npm test

# Test with coverage
npm run test:coverage

# Final build check
npm run build
```

Expected: TypeScript passes with 0 errors. ESLint passes. All frontend tests pass. Build completes successfully.

---

## Stage 16 — Full Integration Smoke Test

This is the final validation before declaring the application ready to run.

### 16.1 — Smoke Test Script

**Create this script in the backend root as `scripts/smoke_test.sh`:**

```bash
#!/bin/bash
# Forensic Council — Full Integration Smoke Test
# Run this script to verify the complete application is working.
# Usage: bash scripts/smoke_test.sh

set -e
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== Forensic Council Smoke Test ===${NC}"

# 1. Check infrastructure
echo -e "\n${YELLOW}[1/8] Checking infrastructure containers...${NC}"
docker compose ps | grep -E "forensic_(redis|qdrant|postgres)" | grep "healthy" | wc -l | xargs -I {} bash -c 'if [ {} -eq 3 ]; then echo -e "${GREEN}All 3 containers healthy${NC}"; else echo -e "${RED}Not all containers healthy — run: docker compose up -d${NC}"; exit 1; fi'

# 2. Check DB schema
echo -e "\n${YELLOW}[2/8] Verifying database schema...${NC}"
uv run python scripts/init_db.py
echo -e "${GREEN}DB schema verified${NC}"

# 3. Run unit tests
echo -e "\n${YELLOW}[3/8] Running unit tests...${NC}"
uv run pytest tests/ -m "unit" -q --tb=short
echo -e "${GREEN}Unit tests passed${NC}"

# 4. Start API server
echo -e "\n${YELLOW}[4/8] Starting API server...${NC}"
uv run python scripts/run_api.py &
API_PID=$!
sleep 3

# 5. Check API health
echo -e "\n${YELLOW}[5/8] Checking API health...${NC}"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs)
if [ "$STATUS" = "200" ]; then
  echo -e "${GREEN}API server responding (HTTP $STATUS)${NC}"
else
  echo -e "${RED}API server not responding (HTTP $STATUS)${NC}"
  kill $API_PID 2>/dev/null
  exit 1
fi

# 6. Run API integration tests
echo -e "\n${YELLOW}[6/8] Running API integration tests...${NC}"
uv run pytest tests/test_api/ -q --tb=short
echo -e "${GREEN}API integration tests passed${NC}"

# 7. Run regression suite
echo -e "\n${YELLOW}[7/8] Running regression suite...${NC}"
uv run pytest tests/test_regression/ -q --tb=short
echo -e "${GREEN}Regression suite passed${NC}"

# 8. Check frontend build
echo -e "\n${YELLOW}[8/8] Verifying frontend build...${NC}"
cd ../agent-council-front-end
npm run build 2>&1 | tail -5
echo -e "${GREEN}Frontend build passed${NC}"
cd ../backend

# Cleanup
kill $API_PID 2>/dev/null

echo -e "\n${GREEN}=== ALL SMOKE TESTS PASSED — Application is ready to run ===${NC}"
```

**Run the smoke test:**
```bash
cd backend
bash scripts/smoke_test.sh
```

---

## HOW TO RUN THE APPLICATION

Once all stages are complete and the smoke test passes, this is the full run procedure.

### Terminal 1 — Infrastructure

```bash
cd backend
docker compose up -d

# Verify all healthy
docker compose ps
```

### Terminal 2 — Backend API Server

```bash
cd backend
uv run python scripts/run_api.py
```

You will see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

API docs are available at: `http://localhost:8000/docs`

### Terminal 3 — Frontend

```bash
cd agent-council-front-end
npm run dev
```

You will see:
```
▲ Next.js 16.1.6
- Local: http://localhost:3000
```

### Using the Application

1. Open `http://localhost:3000` in your browser
2. Navigate to the **Evidence** page
3. Upload a digital evidence file (JPEG, PNG, MP4, WAV, or MP3, max 50MB)
4. Enter a case ID and investigator ID
5. Click **Analyse** — the five agent cards will animate with live findings from the WebSocket
6. If a HITL checkpoint triggers, a modal will appear — review and respond
7. When all agents complete, you are redirected to the **Result** page
8. The full forensic report is displayed with all sections
9. Export as JSON using the export button

### Development Mode Run (Fast — Mocked LLM)

For rapid testing without real LLM inference:

```bash
# In your .env, set:
OPENAI_API_KEY=sk-mock-key-for-testing
APP_ENV=development

# The ReAct loop engine should fall back to deterministic mock responses
# when the LLM key is invalid, producing stub findings rapidly.
# (Verify this behaviour was built correctly in Stage 4's MockLLM fixture.)
```

---

## QUALITY GATES SUMMARY

These must all pass before the application is considered complete.

| Gate | Command | Expected Result |
|---|---|---|
| Frontend build | `npm run build` | 0 errors, 0 warnings |
| TypeScript types | `npm run type-check` | 0 type errors |
| Frontend lint | `npm run lint` | 0 lint errors |
| Frontend tests | `npm test` | All tests pass |
| Backend unit tests | `uv run pytest -m unit -v` | All tests pass |
| Backend integration | `uv run pytest -m integration -v` | All tests pass |
| Backend regression | `uv run pytest -m regression -v` | All tests pass |
| Coverage threshold | `uv run pytest --cov-fail-under=75` | ≥ 75% coverage |
| Chain tamper check | Regression test 1 | `broken_at` correctly identifies corrupted entry |
| Signing verifiable | Stage 11 test | Report signature verifiable with Arbiter public key |
| API health | `curl localhost:8000/docs` | HTTP 200 |
| E2E pipeline | Integration test | Signed report returned with all required sections |
| Full smoke test | `bash scripts/smoke_test.sh` | ALL SMOKE TESTS PASSED |

---

## KNOWN LIMITATIONS AT THIS BUILD STAGE

These are documented gaps that are known and acceptable for the current prototype. They must be addressed before any real investigative use.

| Limitation | Location | Production Replacement |
|---|---|---|
| `face_swap_detect` is a frequency heuristic | `tools/video_tools.py` | FaceForensics++ trained model |
| Calibration uses stub sigmoid curves | `core/calibration.py` | Fit against real benchmark datasets |
| Episodic memory uses random vectors in tests | `core/episodic_memory.py` | Real sentence embeddings (e.g., `sentence-transformers`) |
| KeyStore is in-memory | `core/signing.py` | Hardware Security Module (HSM) or cloud KMS |
| Evidence storage is local filesystem | `infra/storage.py` | S3 with Object Lock or IPFS |
| No PDF report export | `api/routes/investigation.py` | ReportLab or WeasyPrint pipeline |
| No reverse image search | `tools/metadata_tools.py` | Google Vision API or SerpAPI integration |
| No astronomical API | `tools/metadata_tools.py` | USNO API or `astropy` library |
| Cross-session data in React state only | `hooks/useForensicData.ts` | Database-backed history endpoint |
| Single-node deployment | `docker-compose.yml` | Kubernetes with GPU node pools |

---

## APPENDIX — ENVIRONMENT VARIABLES FINAL CHECKLIST

Before running the application, verify all values in your `.env` are set:

```bash
# Required for application to start
OPENAI_API_KEY           ← must be a real OpenAI key for real agent reasoning
POSTGRES_PASSWORD        ← change from default 'forensic_pass'
SIGNING_KEY              ← generate: python -c "import secrets; print(secrets.token_hex(32))"

# Required to match docker-compose port mapping
REDIS_PORT=6380          ← not 6379 — docker-compose maps external port 6380

# Required for frontend
# In agent-council-front-end/.env.local:
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## APPENDIX — GIT CHECKPOINT REFERENCE

```
Stage 0  complete → git commit -m "Stage 0: Infrastructure skeleton — X tests passing"
Stage 1  complete → git commit -m "Stage 1: Signing + CustodyLogger — X tests passing"
Stage 2  complete → git commit -m "Stage 2: Evidence versioning — X tests passing"
Stage 3  complete → git commit -m "Stage 3: Dual-layer memory — X tests passing"
Stage 4  complete → git commit -m "Stage 4: ReAct engine + HITL — X tests passing"
Stage 5  complete → git commit -m "Stage 5: ForensicAgent base + self-reflection — X tests passing"
Stage 6  complete → git commit -m "Stage 6: Image + Metadata tools — X tests passing"
Stage 7  complete → git commit -m "Stage 7: Audio + Video tools — X tests passing"
Stage 8  complete → git commit -m "Stage 8: Inter-agent bus — X tests passing"
Stage 9  complete → git commit -m "Stage 9: Calibration layer — X tests passing"
Stage 10 complete → git commit -m "Stage 10: Adversarial robustness — X tests passing"
Stage 11 complete → git commit -m "Stage 11: Arbiter + Report — X tests passing"
Stage 12 complete → git commit -m "Stage 12: E2E integration — X tests passing"
Stage 13 complete → git commit -m "Stage 13: FastAPI server — X tests passing"
Stage 14 complete → git commit -m "Stage 14: Frontend API integration — build passing"
Stage 15 complete → git commit -m "Stage 15: Regression suite + quality checks — all gates green"
Stage 16 complete → git commit -m "Stage 16: Smoke test passing — application ready"
```

---

*Forensic Council Completion Guide v1.0*
*Covers backend Stages 0–12 (existing roadmap) + Stages 13–16 (API layer, frontend integration, regression testing, smoke test)*
*All architecture principles from Forensic_Council_Architecture.md v3.0 remain in force throughout.*
