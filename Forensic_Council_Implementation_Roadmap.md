# Forensic Council — Kilo Code Implementation Roadmap
## Vibe Coding Build Guide · Precision Test Gates · Stage Completion Reports

---

## How to Use This Document

This roadmap is structured as a sequence of **Kilo Code task prompts**. Each stage gives you:

1. **Exact prompt to paste into Kilo Code** — copy verbatim into the chat/task input
2. **What Kilo Code will build** — so you know what to expect
3. **Test Gate** — commands to run in terminal to confirm the build is correct before proceeding
4. **Stage Completion Report prompt** — paste this at end of each stage so Kilo Code reports back what it built and what remains

### Tool Setup (Before Stage 0)

| Tool | Purpose | Install |
|---|---|---|
| **Kilo Code** | Primary AI coding agent (VS Code extension) | VS Code → Extensions → search "Kilo Code" |
| **Python 3.11+** | Runtime | `brew install python` / `apt install python3.11` |
| **Docker + Docker Compose** | Redis, Qdrant, Postgres containers | https://docker.com |
| **Git** | Version control for stage checkpoints | pre-installed or `brew install git` |
| **uv** | Fast Python package manager | `pip install uv` |

### Project Root

All stages build inside one project folder:
```
mkdir forensic_council && cd forensic_council
git init
code .     ← opens VS Code with Kilo Code
```

---

## Stage 0 — Project Skeleton & Infrastructure

### 0.1 — Kilo Code Prompt

```
Create a Python project called forensic_council with the following structure:

forensic_council/
├── pyproject.toml          (uv-managed, Python 3.11)
├── docker-compose.yml      (Redis, Qdrant, PostgreSQL services)
├── .env.example            (all env vars templated, no real values)
├── README.md               (project overview)
├── core/
│   ├── __init__.py
│   ├── config.py           (Pydantic Settings loading from .env)
│   ├── logging.py          (structured JSON logger)
│   └── exceptions.py       (ForensicCouncilBaseException hierarchy)
├── infra/
│   ├── __init__.py
│   ├── redis_client.py     (async Redis client wrapper)
│   ├── qdrant_client.py    (Qdrant async client wrapper)
│   ├── postgres_client.py  (asyncpg connection pool wrapper)
│   └── storage.py          (immutable file storage abstraction — local filesystem stub)
├── tests/
│   ├── __init__.py
│   ├── conftest.py         (pytest fixtures: test Redis, Qdrant, Postgres connections)
│   └── test_infra/
│       ├── __init__.py
│       ├── test_redis.py
│       ├── test_qdrant.py
│       └── test_postgres.py
└── scripts/
    └── init_db.py          (creates PostgreSQL schema for chain-of-custody log table)

Dependencies to include in pyproject.toml:
- langgraph>=0.2
- langchain>=0.3
- langchain-openai
- redis[asyncio]
- qdrant-client[async]
- asyncpg
- pydantic>=2
- pydantic-settings
- cryptography
- python-dotenv
- pytest
- pytest-asyncio
- pytest-cov
- httpx

docker-compose.yml must define:
- redis: redis:7-alpine, port 6379
- qdrant: qdrant/qdrant:latest, ports 6333/6334
- postgres: postgres:16-alpine, port 5432, db=forensic_council

PostgreSQL chain_of_custody table schema (in init_db.py):
CREATE TABLE IF NOT EXISTS chain_of_custody (
  entry_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  entry_type      VARCHAR(64) NOT NULL,
  agent_id        VARCHAR(64) NOT NULL,
  session_id      UUID NOT NULL,
  timestamp_utc   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  content         JSONB NOT NULL,
  content_hash    VARCHAR(64) NOT NULL,
  signature       TEXT NOT NULL,
  prior_entry_ref VARCHAR(64)
);
CREATE INDEX idx_coc_session ON chain_of_custody(session_id);
CREATE INDEX idx_coc_agent ON chain_of_custody(agent_id);

All client wrappers must:
- Support async context managers
- Log connection events via core/logging.py
- Raise typed ForensicCouncilBaseException subclasses on failure
```

---

### 0.2 — Test Gate (Run These Commands)

```bash
# Start infrastructure
docker compose up -d

# Install dependencies
uv sync

# Run infra tests
uv run pytest tests/test_infra/ -v

# Verify DB schema
uv run python scripts/init_db.py

# Check all containers healthy
docker compose ps
```

**Expected results:**
- All 3 containers: `healthy`
- `test_redis.py` — PASSED (ping, set, get, delete)
- `test_qdrant.py` — PASSED (create collection, upsert, query)
- `test_postgres.py` — PASSED (insert row, select row from chain_of_custody)
- `init_db.py` — prints `Schema initialized successfully`

---

### 0.3 — Stage Completion Report Prompt

```
Stage 0 is complete. Please provide a structured completion report with exactly these sections:

BUILT IN THIS STAGE:
- List every file created with a one-line description

TESTS PASSING:
- List each test file and number of tests passing

INFRASTRUCTURE STATUS:
- List each Docker service and its health status

WHAT REMAINS (next stage):
- One sentence summary of Stage 1

OUTPUT this report as plain text I can paste into my project CHANGELOG.md
```

---

## Stage 1 — Cryptographic Signing & Chain-of-Custody Logger

### 1.1 — Kilo Code Prompt

```
In the forensic_council project, build the cryptographic signing and chain-of-custody logging system.

Create the following files:

core/signing.py
- AgentKeyPair: generates and stores ECDSA P-256 key pairs per agent identity
- sign_content(agent_id: str, content: dict) -> SignedEntry
  - computes SHA-256 hash of JSON-serialized content
  - signs (content_hash + timestamp_utc) with agent's private key
  - returns SignedEntry dataclass: {content, content_hash, signature, agent_id, timestamp_utc}
- verify_entry(entry: SignedEntry) -> bool
  - recomputes hash, verifies signature with agent's public key
  - returns False (never raises) on any verification failure
- KeyStore: dict-backed in-memory store for agent key pairs (stub for HSM in production)
  - get_or_create(agent_id: str) -> AgentKeyPair

core/custody_logger.py
- CustodyLogger: async class backed by postgres_client
  - async log_entry(agent_id, session_id, entry_type, content) -> UUID
    - signs content via signing.py
    - fetches prior_entry_ref (last entry for this session)
    - inserts signed row into chain_of_custody table
    - returns entry_id UUID
  - async get_session_chain(session_id: UUID) -> list[ChainEntry]
    - returns all entries for session ordered by timestamp
  - async verify_chain(session_id: UUID) -> ChainVerificationReport
    - verifies every signature and every prior_entry_ref link
    - returns report: {session_id, total_entries, valid, broken_at: entry_id | None}

Entry types enum in core/custody_logger.py:
THOUGHT, ACTION, OBSERVATION, TOOL_CALL, INTER_AGENT_CALL,
HITL_CHECKPOINT, HUMAN_INTERVENTION, MEMORY_READ, MEMORY_WRITE,
ARTIFACT_VERSION, CALIBRATION, SELF_REFLECTION, FINAL_FINDING,
TRIBUNAL_JUDGMENT, REPORT_SIGNED

Tests to create in tests/test_core/test_signing.py:
- test_sign_and_verify_valid_entry
- test_tampered_content_fails_verification
- test_tampered_signature_fails_verification
- test_different_agent_key_fails_verification
- test_keystore_returns_same_key_for_same_agent

Tests to create in tests/test_core/test_custody_logger.py:
- test_log_single_entry_returns_uuid
- test_chain_links_entries_by_prior_ref
- test_verify_chain_passes_clean_chain
- test_verify_chain_detects_tampered_entry (manually corrupt a db row, verify report shows broken_at)
- test_get_session_chain_returns_ordered_entries
```

---

### 1.2 — Test Gate

```bash
uv run pytest tests/test_core/ -v --cov=core --cov-report=term-missing
```

**Expected results:**
- 10 tests, all PASSED
- Coverage on `core/signing.py` ≥ 90%
- Coverage on `core/custody_logger.py` ≥ 85%
- Tamper detection test must explicitly show `broken_at` is not None

---

### 1.3 — Stage Completion Report Prompt

```
Stage 1 complete. Provide completion report:

BUILT IN THIS STAGE:
- Every file created or modified

TESTS PASSING:
- Each test name and status

COVERAGE:
- Per-file coverage percentages

SIGNING ARCHITECTURE VERIFIED:
- Confirm tamper detection test explicitly catches a corrupted chain entry

WHAT REMAINS:
- One sentence for Stage 2
```

---

## Stage 2 — Evidence Ingestion & Versioning System

### 2.1 — Kilo Code Prompt

```
Build the evidence ingestion and immutable versioning system for forensic_council.

Create:

core/evidence.py
- EvidenceArtifact: Pydantic model
  {artifact_id: UUID, parent_id: UUID | None, root_id: UUID,
   artifact_type: str, file_path: str, content_hash: str,
   action: str, agent_id: str, session_id: UUID, timestamp_utc: datetime,
   metadata: dict}
- ArtifactType enum: ORIGINAL, ELA_OUTPUT, ROI_CROP, AUDIO_SEGMENT,
  VIDEO_FRAME_WINDOW, METADATA_EXPORT, STEGANOGRAPHY_SCAN,
  CODEC_FINGERPRINT, OPTICAL_FLOW_HEATMAP, CALIBRATION_OUTPUT

infra/evidence_store.py
- EvidenceStore: manages immutable artifact storage
  - async ingest(file_path: str, session_id: UUID, agent_id: str) -> EvidenceArtifact
    - computes SHA-256 of file content
    - copies file to immutable storage dir (infra/storage/{root_id}/)
    - creates root EvidenceArtifact (parent_id=None)
    - logs ARTIFACT_VERSION entry to CustodyLogger
    - returns EvidenceArtifact
  - async create_derivative(parent: EvidenceArtifact, data: bytes,
      artifact_type: ArtifactType, action: str, agent_id: str,
      metadata: dict = {}) -> EvidenceArtifact
    - writes data to storage under same root_id dir
    - creates child EvidenceArtifact with parent_id set
    - logs ARTIFACT_VERSION entry to CustodyLogger
    - returns EvidenceArtifact
  - async get_version_tree(root_id: UUID) -> VersionTree
    - returns all artifacts under root_id as tree structure
  - async verify_artifact_integrity(artifact: EvidenceArtifact) -> bool
    - recomputes hash of stored file, compares to artifact.content_hash

Add PostgreSQL table to init_db.py:
CREATE TABLE IF NOT EXISTS evidence_artifacts (
  artifact_id   UUID PRIMARY KEY,
  parent_id     UUID REFERENCES evidence_artifacts(artifact_id),
  root_id       UUID NOT NULL,
  artifact_type VARCHAR(64) NOT NULL,
  file_path     TEXT NOT NULL,
  content_hash  VARCHAR(64) NOT NULL,
  action        TEXT NOT NULL,
  agent_id      VARCHAR(64) NOT NULL,
  session_id    UUID NOT NULL,
  timestamp_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  metadata      JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_ev_root ON evidence_artifacts(root_id);
CREATE INDEX idx_ev_session ON evidence_artifacts(session_id);

Tests in tests/test_infra/test_evidence_store.py:
- test_ingest_creates_root_artifact_with_correct_hash
- test_ingest_copies_file_to_immutable_storage
- test_create_derivative_links_to_parent
- test_version_tree_shows_parent_child_relationship
- test_version_tree_three_levels_deep
- test_integrity_check_passes_untouched_file
- test_integrity_check_fails_modified_file (manually edit stored file)
- test_ingest_logs_artifact_version_to_custody_logger
- test_derivative_logs_artifact_version_to_custody_logger
```

---

### 2.2 — Test Gate

```bash
uv run pytest tests/test_infra/test_evidence_store.py -v
# Verify immutable storage directory created
ls -la infra/storage/
# Run full test suite to confirm no regressions
uv run pytest tests/ -v
```

**Expected:** All 9 evidence tests pass. No regressions in Stage 0/1 tests.

---

### 2.3 — Stage Completion Report Prompt

```
Stage 2 complete. Report:

BUILT:
- Files created/modified

TESTS:
- All test names and pass/fail

EVIDENCE TREE VERIFIED:
- Confirm test_version_tree_three_levels_deep passes and describe what it tested

INTEGRITY CHECK VERIFIED:
- Confirm tamper detection on modified file works

WHAT REMAINS: Stage 3 summary
```

---

## Stage 3 — Working Memory & Episodic Memory

### 3.1 — Kilo Code Prompt

```
Build the dual-layer memory architecture for forensic_council.

Create:

core/working_memory.py
- TaskStatus enum: PENDING, IN_PROGRESS, COMPLETE, BLOCKED
- Task: Pydantic model {task_id: UUID, description: str, status: TaskStatus,
    result_ref: str | None, blocked_reason: str | None}
- WorkingMemory: async class backed by Redis
  - async initialize(session_id: UUID, agent_id: str, tasks: list[str]) -> None
    - writes initial task list to Redis key: wm:{session_id}:{agent_id}
    - logs MEMORY_WRITE to CustodyLogger
  - async update_task(session_id, agent_id, task_id, status, result_ref=None, blocked_reason=None)
    - updates task in Redis
    - logs MEMORY_WRITE to CustodyLogger
  - async get_state(session_id, agent_id) -> WorkingMemoryState
    - returns full task list with statuses
    - logs MEMORY_READ to CustodyLogger
  - async serialize_to_json(session_id, agent_id) -> str
    - full JSON dump for HITL checkpoint persistence
  - async restore_from_json(session_id, agent_id, json_str) -> None
    - restores state from JSON (used on HITL resume)
  - async clear(session_id, agent_id) -> None
    - deletes Redis keys for session end

- WorkingMemoryState: Pydantic model
  {session_id, agent_id, tasks: list[Task], current_iteration: int,
   iteration_ceiling: int, hitl_state: str | None}

core/episodic_memory.py
- ForensicSignatureType enum: DEVICE_FINGERPRINT, METADATA_PATTERN,
    OBJECT_DETECTION, AUDIO_ARTIFACT, VIDEO_ARTIFACT, MANIPULATION_SIGNATURE
- EpisodicEntry: Pydantic model
  {entry_id: UUID, case_id: str, agent_id: str, session_id: UUID,
   signature_type: ForensicSignatureType, finding_type: str,
   confidence: float, summary: str, timestamp_utc: datetime}
- EpisodicMemory: async class backed by Qdrant
  - Collection name: "forensic_episodes"
  - Vector size: 768 (stub — use random vector in tests, real embeddings in production)
  - async store(entry: EpisodicEntry, embedding: list[float]) -> None
    - upserts to Qdrant with entry as payload
    - logs MEMORY_WRITE to CustodyLogger
  - async query(query_embedding: list[float], signature_type: ForensicSignatureType,
      top_k: int = 5) -> list[EpisodicEntry]
    - filtered vector search by signature_type
    - logs MEMORY_READ to CustodyLogger
    - returns list of EpisodicEntry
  - async get_by_case(case_id: str) -> list[EpisodicEntry]
    - retrieves all entries for a case_id via payload filter

Tests in tests/test_core/test_working_memory.py:
- test_initialize_creates_all_tasks_as_pending
- test_update_task_to_in_progress
- test_update_task_to_complete_with_result_ref
- test_update_task_to_blocked_with_reason
- test_get_state_returns_current_status
- test_serialize_restore_roundtrip_preserves_state
- test_clear_removes_redis_keys
- test_all_memory_operations_log_to_custody_logger

Tests in tests/test_core/test_episodic_memory.py:
- test_store_entry_persists_to_qdrant
- test_query_returns_matching_entries
- test_query_filters_by_signature_type
- test_get_by_case_returns_all_case_entries
- test_store_logs_memory_write_to_custody_logger
- test_query_logs_memory_read_to_custody_logger
```

---

### 3.2 — Test Gate

```bash
uv run pytest tests/test_core/test_working_memory.py tests/test_core/test_episodic_memory.py -v

# Verify Redis keys exist after test run (before clear)
docker exec -it forensic_council_redis_1 redis-cli KEYS "wm:*"

# Verify Qdrant collection created
curl http://localhost:6333/collections/forensic_episodes
```

**Expected:** 14 tests pass. Redis keys visible mid-test. Qdrant returns collection info with vector count > 0.

---

### 3.3 — Stage Completion Report Prompt

```
Stage 3 complete. Report:

BUILT: files created/modified
TESTS: all test names and status
MEMORY VERIFIED:
- Confirm serialize/restore roundtrip test description
- Confirm custody logger receives both READ and WRITE events from memory operations
WHAT REMAINS: Stage 4 summary
```

---

## Stage 4 — ReAct Loop Engine & HITL Checkpoint System

### 4.1 — Kilo Code Prompt

```
Build the core ReAct loop engine and HITL checkpoint system for forensic_council.
This is the most critical architectural component.

Create:

core/react_loop.py

ReActStep: Pydantic model
  {step_type: Literal["THOUGHT","ACTION","OBSERVATION"],
   content: str, tool_name: str | None, tool_input: dict | None,
   tool_output: dict | None, iteration: int, timestamp_utc: datetime}

HITLCheckpointReason enum:
  ITERATION_CEILING_50PCT, CONTESTED_FINDING, TOOL_UNAVAILABLE,
  SEVERITY_THRESHOLD_BREACH, TRIBUNAL_ESCALATION

HITLCheckpointState: Pydantic model
  {checkpoint_id: UUID, agent_id: str, session_id: UUID,
   reason: HITLCheckpointReason, current_finding_summary: str,
   paused_at_iteration: int, investigator_brief: str,
   status: Literal["PAUSED","RESUMED","OVERRIDDEN","TERMINATED"]}

HumanDecision: Pydantic model
  {decision_type: Literal["APPROVE","REDIRECT","OVERRIDE","TERMINATE","ESCALATE"],
   investigator_id: str, notes: str, override_finding: dict | None}

ReActLoopEngine: async class
  Constructor: (agent_id, session_id, iteration_ceiling, working_memory,
                custody_logger, evidence_store)

  async run(initial_thought: str, tool_registry: ToolRegistry) -> ReActLoopResult
    - Starts loop from initial_thought
    - Each iteration:
        1. LLM generates next step (THOUGHT/ACTION/OBSERVATION)
        2. If ACTION: calls tool via ToolRegistry
        3. Logs each step as THOUGHT/ACTION/OBSERVATION to CustodyLogger
        4. Updates working memory with current iteration
        5. Checks HITL trigger conditions after each step
        6. If HITL triggered: pauses loop, emits checkpoint, awaits resume signal
    - Loop ends when: LLM signals completion, ceiling reached, or TERMINATE received
    - Returns ReActLoopResult

  async check_hitl_triggers(state: WorkingMemoryState) -> HITLCheckpointReason | None
    - Returns reason if any trigger condition met, else None
    - Trigger at 50% of iteration_ceiling without COMPLETE task
    - Trigger on CONTESTED_FINDING in current findings
    - Trigger on SEVERITY_THRESHOLD items in current findings

  async pause_for_hitl(reason: HITLCheckpointReason, brief: str) -> HITLCheckpointState
    - Serializes working memory to JSON
    - Logs HITL_CHECKPOINT to CustodyLogger
    - Stores checkpoint in Redis: hitl:{session_id}:{agent_id}
    - Returns HITLCheckpointState with status=PAUSED

  async resume_from_hitl(checkpoint_id: UUID, decision: HumanDecision) -> None
    - Logs HUMAN_INTERVENTION to CustodyLogger (signed with human note)
    - If REDIRECT: injects context into working memory
    - If OVERRIDE: logs override finding as FINAL_FINDING
    - If TERMINATE: sets loop termination flag
    - Restores working memory from serialized JSON
    - Clears checkpoint from Redis

ReActLoopResult: Pydantic model
  {session_id, agent_id, completed: bool, terminated_by_human: bool,
   findings: list[AgentFinding], hitl_checkpoints: list[HITLCheckpointState],
   total_iterations: int, react_chain: list[ReActStep]}

AgentFinding: Pydantic model
  {finding_id: UUID, agent_id: str, finding_type: str,
   confidence_raw: float, calibrated: bool,
   status: Literal["CONFIRMED","CONTESTED","INCONCLUSIVE","INCOMPLETE"],
   robustness_caveat: bool, robustness_caveat_detail: str | None,
   evidence_refs: list[UUID], reasoning_summary: str}

core/tool_registry.py
- Tool: Pydantic model {name: str, description: str, available: bool}
- ToolResult: Pydantic model {tool_name: str, success: bool, output: dict,
    error: str | None, unavailable: bool}
- ToolRegistry: manages available tools per agent
  - register(tool_name: str, handler: Callable) -> None
  - async call(tool_name: str, input: dict, agent_id: str,
      session_id: UUID, custody_logger: CustodyLogger) -> ToolResult
    - Checks availability before calling
    - If unavailable: returns ToolResult(unavailable=True) — NEVER raises
    - Logs TOOL_CALL to CustodyLogger (input + output + availability status)
  - set_unavailable(tool_name: str) -> None (for testing graceful degradation)

Tests in tests/test_core/test_react_loop.py:
Use a mock LLM that returns scripted THOUGHT/ACTION/OBSERVATION sequences.

- test_loop_runs_thought_action_observation_cycle
- test_loop_logs_every_step_to_custody_logger
- test_loop_stops_at_iteration_ceiling
- test_hitl_triggered_at_50pct_ceiling_without_finding
- test_hitl_pause_serializes_working_memory
- test_hitl_resume_approve_continues_loop
- test_hitl_resume_terminate_stops_loop
- test_hitl_resume_override_logs_human_judgment
- test_tool_unavailable_does_not_crash_loop
- test_tool_unavailable_logged_as_incomplete_finding

Tests in tests/test_core/test_tool_registry.py:
- test_register_and_call_tool
- test_unavailable_tool_returns_graceful_result
- test_tool_call_logged_to_custody_logger
- test_tool_call_input_and_output_both_logged
```

---

### 4.2 — Test Gate

```bash
uv run pytest tests/test_core/test_react_loop.py tests/test_core/test_tool_registry.py -v

# Confirm HITL pause stores checkpoint in Redis
# Run the HITL pause test in isolation and inspect Redis
uv run pytest tests/test_core/test_react_loop.py::test_hitl_pause_serializes_working_memory -v -s

# Run full test suite — no regressions
uv run pytest tests/ -v --tb=short
```

**Expected:** 14 tests pass. Full suite green. HITL test output shows Redis key `hitl:{uuid}:{agent_id}` was created and contains serialized working memory.

---

### 4.3 — Stage Completion Report Prompt

```
Stage 4 complete. Report:

BUILT: files created/modified
TESTS: all test names and status
REACT LOOP VERIFIED:
- Confirm THOUGHT/ACTION/OBSERVATION cycle test description
- Confirm HITL pause/resume test proves working memory survives pause
GRACEFUL DEGRADATION VERIFIED:
- Confirm unavailable tool test does not raise exception
FULL SUITE STATUS: X tests passing, 0 failing
WHAT REMAINS: Stage 5 summary
```

---

## Stage 5 — Agent Base Class & Self-Reflection System

### 5.1 — Kilo Code Prompt

```
Build the ForensicAgent base class and self-reflection system.
Every specialist agent (1-5) will extend this base class.

Create:

agents/base_agent.py

SelfReflectionReport: Pydantic model
  {all_tasks_complete: bool, incomplete_tasks: list[str],
   overconfident_findings: list[str], untreated_absences: list[str],
   deprioritized_avenues: list[str], court_defensible: bool,
   reflection_notes: str}

ForensicAgent: abstract async base class
  Constructor: (agent_id, session_id, evidence_artifact, config,
                working_memory, episodic_memory, custody_logger, evidence_store)

  Abstract properties (must override):
    - agent_name: str
    - task_decomposition: list[str]   ← hardcoded per agent
    - iteration_ceiling: int

  Abstract methods (must override):
    - async build_tool_registry() -> ToolRegistry
    - async build_initial_thought() -> str

  Concrete methods (shared by all agents):

  async run_investigation() -> AgentFinding list
    1. Initialize working memory with task_decomposition
    2. Log session start to CustodyLogger
    3. Build tool registry (calls abstract method)
    4. Check tool availability — log any unavailable tools
    5. Build initial thought (calls abstract method)
    6. Run ReActLoopEngine
    7. Run self_reflection_pass()
    8. Return findings

  async self_reflection_pass(findings: list[AgentFinding]) -> SelfReflectionReport
    Uses 5 structured reflection prompts:
      RT1: All tasks complete?
      RT2: Overconfident findings?
      RT3: Absences treated as signals?
      RT4: Deprioritized avenues?
      RT5: Confidence court-defensible?
    Logs SELF_REFLECTION to CustodyLogger
    If reflection reveals issues: re-opens loop for targeted re-examination
    Returns SelfReflectionReport

  async query_episodic_memory(signature_type, query_embedding) -> list[EpisodicEntry]
    Wraps episodic_memory.query() with custody logging

  async store_episodic_finding(entry: EpisodicEntry, embedding) -> None
    Wraps episodic_memory.store() with custody logging

  async flag_hitl(reason, brief) -> None
    Delegates to loop engine's pause_for_hitl

agents/agent1_image.py — Agent 1 stub (extend ForensicAgent)
  - agent_name = "Agent1_ImageIntegrity"
  - iteration_ceiling = 20
  - task_decomposition = [exact 8 tasks from architecture doc]
  - build_tool_registry(): registers stub tools:
      ela_full_image, roi_extract, jpeg_ghost_detect,
      frequency_domain_analysis, file_hash_verify,
      adversarial_robustness_check, sensor_db_query
  - build_initial_thought(): returns Agent 1's opening thought string

agents/agent2_audio.py — Agent 2 stub (same pattern, 10 tasks)
agents/agent3_object.py — Agent 3 stub (9 tasks)
agents/agent4_video.py — Agent 4 stub (9 tasks)
agents/agent5_metadata.py — Agent 5 stub (11 tasks)

All tool handlers in stubs return: {"status": "stub_response", "tool": tool_name}
This is enough to test the base class — real tool implementations come in Stage 6+.

Tests in tests/test_agents/test_base_agent.py:
- test_agent_initializes_working_memory_with_task_list
- test_agent_logs_session_start_to_custody
- test_self_reflection_runs_all_5_reflection_prompts
- test_self_reflection_logged_to_custody
- test_self_reflection_incomplete_tasks_flagged
- test_full_investigation_run_returns_findings
- test_all_five_agent_stubs_instantiate_without_error
- test_all_five_agents_have_correct_task_counts (8, 10, 9, 9, 11)
```

---

### 5.2 — Test Gate

```bash
uv run pytest tests/test_agents/ -v

# Verify all 5 agents have correct task counts
uv run python -c "
from agents.agent1_image import Agent1Image
from agents.agent2_audio import Agent2Audio
from agents.agent3_object import Agent3Object
from agents.agent4_video import Agent4Video
from agents.agent5_metadata import Agent5Metadata
for cls in [Agent1Image, Agent2Audio, Agent3Object, Agent4Video, Agent5Metadata]:
    print(f'{cls.__name__}: {len(cls.task_decomposition)} tasks')
"

uv run pytest tests/ -v --tb=short
```

**Expected:** Prints exact task counts (8, 10, 9, 9, 11). All tests green.

---

### 5.3 — Stage Completion Report Prompt

```
Stage 5 complete. Report:

BUILT: files created/modified
TESTS: all test names and status
TASK COUNTS: Confirm exact task count per agent matches architecture doc
SELF-REFLECTION VERIFIED: Confirm all 5 reflection prompts logged
FULL SUITE: X tests, 0 failing
WHAT REMAINS: Stage 6 summary
```

---

## Stage 6 — Real Tool Implementations (Agents 1 & 5)

### 6.1 — Kilo Code Prompt

```
Implement real forensic tool handlers for Agent 1 (Image Integrity) and
Agent 5 (Metadata Analysis). These are the lowest-dependency agents to validate
real tool integration before tackling audio/video.

Add dependencies to pyproject.toml:
- Pillow
- numpy
- scipy
- exiftool-python (wrapper for ExifTool CLI — must be installed: brew install exiftool)
- piexif
- imagehash

Create tools/image_tools.py:

ela_full_image(artifact: EvidenceArtifact) -> dict
  - Opens image with Pillow
  - Saves at quality=95, reloads, computes pixel diff → ELA map
  - Returns: {ela_map_array: list, max_anomaly: float, anomaly_regions: list[BoundingBox]}
  - Creates derivative artifact (ArtifactType.ELA_OUTPUT) via evidence_store
  - Raises ToolUnavailableError if file cannot be opened

roi_extract(artifact: EvidenceArtifact, bounding_box: dict) -> dict
  - Crops image to bounding_box {x, y, w, h}
  - Returns: {roi_artifact: EvidenceArtifact} (new derivative)
  - Creates derivative artifact (ArtifactType.ROI_CROP)

jpeg_ghost_detect(artifact: EvidenceArtifact) -> dict
  - Saves image at multiple quality levels (50,60,70,80,90)
  - Computes variance map across quality levels
  - Returns: {ghost_detected: bool, confidence: float, ghost_regions: list[BoundingBox]}

file_hash_verify(artifact: EvidenceArtifact, evidence_store: EvidenceStore) -> dict
  - Calls evidence_store.verify_artifact_integrity(artifact)
  - Returns: {hash_matches: bool, original_hash: str, current_hash: str}

Create tools/metadata_tools.py:

exif_extract(artifact: EvidenceArtifact) -> dict
  - Runs exiftool on file path
  - Returns: {present_fields: dict, absent_fields: list[str], device_model: str | None}
  - absent_fields computed by comparing returned fields to EXPECTED_EXIF_FIELDS constant
  - EXPECTED_EXIF_FIELDS = [full list of standard JPEG/PNG EXIF fields]

gps_timezone_validate(gps_lat: float, gps_lon: float, timestamp_utc: str) -> dict
  - Uses timezonefinder library to get timezone at GPS coords
  - Validates timestamp is plausible (not future, timezone consistent)
  - Returns: {timezone: str, plausible: bool, offset_hours: float}

file_hash_verify — reuse from image_tools.py

steganography_scan(artifact: EvidenceArtifact) -> dict
  - Uses LSB analysis via numpy to detect statistical anomalies in pixel LSBs
  - Returns: {stego_suspected: bool, confidence: float, method: str}

Add to pyproject.toml: timezonefinder

Update agents/agent1_image.py:
  Replace stub tool handlers with real implementations from tools/image_tools.py

Update agents/agent5_metadata.py:
  Replace stub tool handlers with real implementations from tools/metadata_tools.py

Tests in tests/test_tools/test_image_tools.py:
Use test fixtures: a known-clean JPEG and a manually spliced JPEG (create programmatically in conftest).
- test_ela_clean_image_returns_low_anomaly
- test_ela_spliced_image_returns_elevated_anomaly_in_splice_region
- test_roi_extract_creates_correct_derivative_artifact
- test_jpeg_ghost_detects_double_compressed_region
- test_file_hash_verify_passes_untouched_file
- test_file_hash_verify_fails_modified_file

Tests in tests/test_tools/test_metadata_tools.py:
Use test fixtures: a JPEG with full EXIF, a JPEG with stripped EXIF.
- test_exif_extract_returns_present_fields
- test_exif_extract_flags_absent_fields_on_stripped_image
- test_gps_timezone_validate_london_coordinates
- test_gps_timezone_validate_flags_implausible_timestamp
- test_steganography_scan_clean_image_returns_false
```

---

### 6.2 — Test Gate

```bash
# Verify exiftool is installed
exiftool -ver

uv run pytest tests/test_tools/ -v

# Run Agent 5 on a real image end-to-end
uv run python -c "
import asyncio
from agents.agent5_metadata import Agent5Metadata
# (use test fixture image path)
print('Agent 5 end-to-end: OK')
"

uv run pytest tests/ -v --tb=short
```

**Expected:** All tool tests pass. ELA test confirms elevated anomaly in splice region (quantitative threshold check, not just boolean).

---

### 6.3 — Stage Completion Report Prompt

```
Stage 6 complete. Report:

BUILT: files
TESTS: all test names and status
ELA VERIFICATION: What anomaly value was detected in the spliced image test?
EXIF VERIFICATION: List the absent fields detected in the stripped EXIF test
FULL SUITE: X tests, 0 failing
WHAT REMAINS: Stage 7 summary
```

---

## Stage 7 — Audio & Video Tool Implementations (Agents 2 & 4)

### 7.1 — Kilo Code Prompt

```
Implement real forensic tool handlers for Agent 2 (Audio) and Agent 4 (Video).

Add to pyproject.toml:
- speechbrain
- pyannote.audio
- librosa
- soundfile
- torch
- torchvision
- opencv-python
- facenet-pytorch (for face detection in deepfake check stub)

Create tools/audio_tools.py:

speaker_diarize(artifact: EvidenceArtifact) -> dict
  - Uses pyannote.audio for speaker diarization
  - Returns: {speaker_count: int, segments: list[{speaker_id, start, end}]}
  - Creates CODEC_FINGERPRINT derivative artifact

anti_spoofing_detect(artifact: EvidenceArtifact, segment: dict | None = None) -> dict
  - Uses SpeechBrain AASIST or RawBoost model on full file or segment
  - Returns: {spoof_detected: bool, confidence: float, model_version: str}

prosody_analyze(artifact: EvidenceArtifact) -> dict
  - Uses librosa to extract pitch, energy, rhythm features
  - Detects statistical discontinuities (anomalous intonation)
  - Returns: {anomalies: list[{timestamp, type, severity}]}

background_noise_consistency(artifact: EvidenceArtifact) -> dict
  - Segments audio, computes noise floor per segment, detects shift points
  - Returns: {shift_points: list[float], consistent: bool}

codec_fingerprint(artifact: EvidenceArtifact) -> dict
  - Uses librosa / ffprobe to detect re-encoding events
  - Returns: {reencoding_events: list[{timestamp, confidence}], codec_chain: list[str]}

Create tools/video_tools.py:

optical_flow_analyze(artifact: EvidenceArtifact) -> dict
  - Uses OpenCV Farneback optical flow on full video
  - Computes per-frame motion vectors, flags statistical outliers
  - Returns: {anomaly_heatmap_artifact: EvidenceArtifact, flagged_frames: list[int]}
  - Creates OPTICAL_FLOW_HEATMAP derivative artifact

frame_window_extract(artifact: EvidenceArtifact, start_frame: int, end_frame: int) -> dict
  - Extracts frame range using OpenCV
  - Returns: {frames_artifact: EvidenceArtifact, frame_count: int}
  - Creates VIDEO_FRAME_WINDOW derivative artifact

frame_consistency_analyze(frames_artifact: EvidenceArtifact) -> dict
  - Computes histogram diff, edge map diff between consecutive frames
  - Returns: {inconsistencies: list[{frame_pair, diff_score}], classification_hint: str}

face_swap_detect(frames_artifact: EvidenceArtifact) -> dict
  - Detects faces per frame using facenet-pytorch
  - Runs frequency-domain analysis on face regions (FFT anomalies typical of GAN faces)
  - Returns: {deepfake_suspected: bool, confidence: float, flagged_frames: list[int]}
  NOTE: This is a heuristic stub — production should use FaceForensics++ trained model.
        Document this clearly in the function docstring.

Update agents/agent2_audio.py — replace stub handlers with real tools
Update agents/agent4_video.py — replace stub handlers with real tools

Tests in tests/test_tools/test_audio_tools.py:
Use test fixtures in conftest: generate clean WAV + WAV with injected splice (silence gap + different noise floor segment).
- test_speaker_diarize_returns_speaker_count
- test_anti_spoofing_passes_clean_audio
- test_prosody_detects_discontinuity_in_spliced_audio
- test_background_noise_consistency_flags_noise_shift
- test_codec_fingerprint_detects_reencoding_event

Tests in tests/test_tools/test_video_tools.py:
Use test fixtures: generate synthetic MP4 in conftest using OpenCV (solid color frames + deliberate frame edit).
- test_optical_flow_generates_heatmap_artifact
- test_optical_flow_flags_abrupt_motion_change
- test_frame_window_extract_creates_derivative
- test_frame_consistency_detects_discontinuity
- test_face_swap_detect_runs_without_error_on_no_face_frame
```

---

### 7.2 — Test Gate

```bash
uv run pytest tests/test_tools/test_audio_tools.py tests/test_tools/test_video_tools.py -v

# Confirm derivative artifacts created for video tests
uv run pytest tests/test_tools/test_video_tools.py::test_optical_flow_generates_heatmap_artifact -v -s

uv run pytest tests/ -v --tb=short
```

**Expected:** All tool tests pass. Optical flow test shows OPTICAL_FLOW_HEATMAP artifact stored in version tree.

---

### 7.3 — Stage Completion Report Prompt

```
Stage 7 complete. Report:

BUILT: files
TESTS: all test names and status
AUDIO SPLICE DETECTION: What discontinuity metric was returned in the noise shift test?
VIDEO ANOMALY: What frame numbers were flagged in the optical flow test?
STUB DOCUMENTATION: Confirm face_swap_detect docstring notes production model requirement
FULL SUITE: X tests, 0 failing
WHAT REMAINS: Stage 8 — Inter-Agent Communication
```

---

## Stage 8 — Inter-Agent Communication Protocol

### 8.1 — Kilo Code Prompt

```
Implement the inter-agent communication protocol with anti-circular dependency enforcement.

Create:

core/inter_agent_bus.py

InterAgentCallType enum: COLLABORATIVE, CHALLENGE

InterAgentCall: Pydantic model
  {call_id: UUID, caller_agent_id: str, callee_agent_id: str,
   call_type: InterAgentCallType,
   payload: {timestamp_ref: str | None, region_ref: dict | None,
             context_finding: dict | None, question: str},
   response: dict | None, status: Literal["PENDING","COMPLETE","FAILED"]}

PERMITTED_CALL_PATHS: dict = {
  "Agent2_Audio": ["Agent4_Video"],
  "Agent4_Video": ["Agent2_Audio"],
  "Agent3_Object": ["Agent1_ImageIntegrity"],
  "Arbiter": ["Agent1_ImageIntegrity", "Agent2_Audio", "Agent3_Object",
               "Agent4_Video", "Agent5_Metadata"]
}

ANTI_CIRCULAR_RULES: Callee cannot re-initiate call to caller on same artifact within same loop.
Agent 3 → Agent 1 is one-way only. Agent 1 may not call Agent 3.
Arbiter challenges are terminal — challenged agent cannot re-challenge.

InterAgentBus: async class
  - async dispatch(call: InterAgentCall, callee_agent: ForensicAgent,
        custody_logger: CustodyLogger) -> dict
    1. Validates call path is in PERMITTED_CALL_PATHS
    2. Validates no circular dependency (checks active call stack)
    3. Logs INTER_AGENT_CALL to BOTH caller and callee custody chains
    4. Invokes callee's handle_inter_agent_call() method
    5. Returns response
    6. Raises PermittedCallViolationError if path not permitted
    7. Raises CircularCallError if circular dependency detected

  - _active_calls: set[tuple] — tracks (caller, callee, artifact_id) in-flight

Add to ForensicAgent base class:
  async handle_inter_agent_call(call: InterAgentCall) -> dict
    Default implementation: runs targeted sub-analysis based on call payload
    Returns findings dict

Update agents/agent2_audio.py:
  In ReAct loop, after flagging timestamp anomaly: dispatch call to Agent 4

Update agents/agent4_video.py:
  In ReAct loop, after flagging visual anomaly: dispatch call to Agent 2

Update agents/agent3_object.py:
  In ReAct loop, after lighting inconsistency: dispatch call to Agent 1

Tests in tests/test_core/test_inter_agent_bus.py:
- test_permitted_call_agent2_to_agent4_succeeds
- test_permitted_call_agent4_to_agent2_succeeds
- test_permitted_call_agent3_to_agent1_succeeds
- test_unpermitted_call_agent1_to_agent3_raises_error
- test_unpermitted_call_agent1_to_agent2_raises_error
- test_circular_call_detected_and_blocked
- test_inter_agent_call_logged_to_both_custody_chains
- test_arbiter_challenge_call_permitted_to_all_agents
- test_arbiter_challenge_cannot_be_re_challenged
```

---

### 8.2 — Test Gate

```bash
uv run pytest tests/test_core/test_inter_agent_bus.py -v

# Verify both custody chains receive the inter-agent call log
uv run pytest tests/test_core/test_inter_agent_bus.py::test_inter_agent_call_logged_to_both_custody_chains -v -s

uv run pytest tests/ -v --tb=short
```

**Expected:** All 9 bus tests pass. Circular call test explicitly raises `CircularCallError`. Both custody chains test confirms 2 INTER_AGENT_CALL entries in DB (one per chain).

---

### 8.3 — Stage Completion Report Prompt

```
Stage 8 complete. Report:

BUILT: files
TESTS: all test names and status
CIRCULAR DEPENDENCY: Confirm exact error type raised and what was attempted
DUAL LOGGING: Confirm both custody chains received inter-agent call log entry
PERMITTED PATHS ENFORCED: List all tested paths and outcomes
FULL SUITE: X tests, 0 failing
WHAT REMAINS: Stage 9 — Confidence Calibration
```

---

## Stage 9 — Confidence Calibration Layer

### 9.1 — Kilo Code Prompt

```
Build the confidence calibration framework with versioned calibration models.

Create:

core/calibration.py

CalibrationMethod enum: PLATT_SCALING, ISOTONIC_REGRESSION, TEMPERATURE_SCALING, RULE_BASED

CalibrationModel: Pydantic model
  {model_id: UUID, agent_id: str, method: CalibrationMethod,
   benchmark_dataset: str, version: str, created_utc: datetime,
   params: dict}   ← stores fitted model parameters as serialized dict

CalibratedConfidence: Pydantic model
  {raw_score: float, calibrated_probability: float, true_positive_rate: float,
   false_positive_rate: float, calibration_model_id: UUID,
   calibration_version: str, benchmark_dataset: str,
   court_statement: str}   ← human-readable admissibility statement

CalibrationLayer: class
  - load_model(agent_id: str, version: str = "latest") -> CalibrationModel
    Loads from local JSON store (infra/calibration_models/{agent_id}/)
  - calibrate(agent_id: str, raw_score: float,
        finding_class: str) -> CalibratedConfidence
    Applies loaded model to raw score
    Returns CalibratedConfidence with auto-generated court_statement:
    "Based on benchmark performance against {dataset}, a model confidence of
     {raw:.2f} in class '{finding_class}' corresponds to a true positive rate
     of {tpr:.1%} with a false positive rate of {fpr:.1%}."
  - fit_stub_model(agent_id: str) -> CalibrationModel
    Generates a stub sigmoid calibration curve for testing
    (production: fit against real benchmark data)

CALIBRATION VERSIONING RULES (enforce in code):
  - CalibrationModel objects are immutable once created
  - Fitting a new model creates a NEW version — never overwrites
  - AgentFinding always embeds calibration_model_id at production time
  - No retroactive re-calibration: if calibration version changes,
    prior findings retain their original calibration_model_id

Update ForensicAgent base class:
  - Add calibration_layer: CalibrationLayer to constructor
  - After self_reflection_pass: run calibrate_all_findings()
  - calibrate_all_findings(): for each CONFIRMED finding, call calibration_layer.calibrate()
    and update finding.calibrated = True, attach CalibratedConfidence
  - Logs CALIBRATION entry to CustodyLogger per finding

Tests in tests/test_core/test_calibration.py:
- test_calibrate_returns_calibrated_confidence_object
- test_court_statement_contains_benchmark_name
- test_court_statement_contains_tpr_and_fpr
- test_calibration_model_versioning_creates_new_version
- test_old_version_model_still_loadable_after_new_version_created
- test_finding_retains_original_calibration_version_id
- test_uncalibrated_finding_flagged_if_calibration_skipped
- test_all_agents_have_stub_calibration_models_loadable
```

---

### 9.2 — Test Gate

```bash
uv run pytest tests/test_core/test_calibration.py -v

# Verify court statement format
uv run python -c "
from core.calibration import CalibrationLayer
cl = CalibrationLayer()
cl.fit_stub_model('Agent1_ImageIntegrity')
result = cl.calibrate('Agent1_ImageIntegrity', 0.89, 'splicing_detected')
print(result.court_statement)
"

uv run pytest tests/ -v --tb=short
```

**Expected:** Court statement printed matches required format exactly. All 8 calibration tests pass.

---

### 9.3 — Stage Completion Report Prompt

```
Stage 9 complete. Report:

BUILT: files
TESTS: all test names and status
COURT STATEMENT: Paste the exact court statement generated by the verification command
VERSIONING VERIFIED: Confirm old version remains loadable after new version created
FINDING IMMUTABILITY: Confirm finding retains original calibration_model_id even when newer model exists
FULL SUITE: X tests, 0 failing
WHAT REMAINS: Stage 10 — Adversarial Robustness Checks
```

---

## Stage 10 — Adversarial Robustness Checks

### 10.1 — Kilo Code Prompt

```
Implement adversarial robustness checks for all five agents.

Create:

core/adversarial.py

RobustnessCaveat: Pydantic model
  {caveat_id: UUID, agent_id: str, evasion_technique: str,
   plausibility: Literal["LOW","MEDIUM","HIGH"],
   detection_basis: str, court_disclosure: str}

AdversarialChecker: class with one method per agent domain:

  check_anti_ela_evasion(image_artifact: EvidenceArtifact) -> list[RobustnessCaveat]
    Tests for:
    - Uniform recompression (checks if JPEG quality levels show unusually flat ELA)
    - Anti-forensic texture synthesis signatures (spatial frequency uniformity)
    Returns caveats with plausibility rating

  check_anti_spoofing_evasion(audio_artifact: EvidenceArtifact) -> list[RobustnessCaveat]
    Tests for:
    - Statistical uniformity in frequency bands (adversarial perturbation signature)
    - Prosody normalization artifacts (unnaturally smooth pitch contour)

  check_object_detection_evasion(image_artifact: EvidenceArtifact) -> list[RobustnessCaveat]
    Tests for:
    - High-frequency pixel patterns in object bounding box (adversarial patch signature)
    - Anomalous texture regularity in detected object region

  check_optical_flow_evasion(video_artifact: EvidenceArtifact) -> list[RobustnessCaveat]
    Tests for:
    - Unusually smooth motion vectors at flagged edit point (flow smoothing applied)
    - Frame blending artifacts at splice boundary

  check_metadata_spoofing(artifact: EvidenceArtifact) -> list[RobustnessCaveat]
    Tests for:
    - Suspiciously complete metadata on low-end device (spoofing pattern)
    - GPS precision exceeding device class capability

Each check must:
  - Return [] (empty list) if no evasion plausible — never return false positives
  - Generate court_disclosure string:
    "A known anti-forensics technique ({technique}) could produce findings
     consistent with this evidence. This does not invalidate the finding but
     must be disclosed to the court."
  - Be integrated as the final ACTION in each agent's ReAct loop before self-reflection

Update all 5 agents:
  Add adversarial_checker: AdversarialChecker to constructor
  Add "Run adversarial robustness check" as final tool in each agent's tool registry
  If caveats returned: set finding.robustness_caveat = True and attach caveat detail

Tests in tests/test_core/test_adversarial.py:
- test_clean_image_returns_no_ela_caveats
- test_uniform_compressed_image_returns_ela_caveat
- test_clean_audio_returns_no_spoofing_caveats
- test_smooth_pitch_contour_audio_returns_caveat
- test_adversarial_patch_image_returns_object_detection_caveat
- test_smooth_optical_flow_at_edit_returns_caveat
- test_complete_metadata_on_low_end_device_returns_spoofing_caveat
- test_caveat_court_disclosure_contains_technique_name
- test_empty_caveat_list_does_not_set_robustness_flag_on_finding
```

---

### 10.2 — Test Gate

```bash
uv run pytest tests/test_core/test_adversarial.py -v

# Verify caveat disclosure format
uv run python -c "
from core.adversarial import AdversarialChecker
# (use a test artifact stub)
print('Adversarial checker imports OK')
"

uv run pytest tests/ -v --tb=short
```

**Expected:** All 9 adversarial tests pass. Full suite green.

---

### 10.3 — Stage Completion Report Prompt

```
Stage 10 complete. Report:

BUILT: files
TESTS: all test names and status
EVASION TECHNIQUES CATALOGUED: List all technique names registered per agent
DISCLOSURE FORMAT: Paste example court_disclosure string generated
FULL SUITE: X tests, 0 failing
WHAT REMAINS: Stage 11 — Council Arbiter
```

---

## Stage 11 — Council Arbiter & Report Generator

### 11.1 — Kilo Code Prompt

```
Build the Council Arbiter — the deliberation, challenge loop, Tribunal, and
court-admissible report generator. This is the synthesis layer.

Create:

agents/arbiter.py

FindingComparison: Pydantic model
  {finding_a: AgentFinding, finding_b: AgentFinding,
   verdict: Literal["AGREEMENT","INDEPENDENT","CONTRADICTION"],
   cross_modal_confirmed: bool}

ChallengeResult: Pydantic model
  {challenge_id: UUID, challenged_agent: str, original_finding: AgentFinding,
   revised_finding: AgentFinding | None, resolved: bool}

TribunalCase: Pydantic model
  {tribunal_id: UUID, agent_a_id: str, agent_b_id: str,
   contradiction: FindingComparison, human_judgment: dict | None,
   resolved: bool}

ForensicReport: Pydantic model (all sections from architecture doc)
  {report_id: UUID, session_id: UUID, case_id: str,
   executive_summary: str,
   per_agent_findings: dict[str, list[AgentFinding]],
   cross_modal_confirmed: list[AgentFinding],
   contested_findings: list[FindingComparison],
   tribunal_resolved: list[TribunalCase],
   incomplete_findings: list[AgentFinding],
   case_linking_flags: list[EpisodicEntry],
   chain_of_custody_log: list[dict],
   evidence_version_trees: list[dict],
   react_chains: dict[str, list[ReActStep]],
   self_reflection_outputs: dict[str, SelfReflectionReport],
   uncertainty_statement: str,
   cryptographic_signature: str,
   report_hash: str,
   signed_utc: datetime}

CouncilArbiter: async class
  Constructor: (session_id, custody_logger, inter_agent_bus, calibration_layer)

  async deliberate(agent_results: dict[str, AgentLoopResult]) -> ForensicReport
    1. Index all findings with confidence scores
    2. Run cross_agent_comparison() — map agreements, independent, contradictions
    3. For each contradiction: run challenge_loop()
    4. For unresolved: trigger_tribunal()
    5. Weight findings — cross-modal confirmed findings auto-elevated
    6. Generate XAI explanation layer
    7. Compile chain-of-custody log
    8. Build ForensicReport
    9. Sign report with arbiter key
    10. Return signed report

  async cross_agent_comparison(all_findings: list[AgentFinding]) -> list[FindingComparison]
    Detects: same evidence region with contradictory conclusions = CONTRADICTION
    Detects: same evidence region, same conclusion from different agents = AGREEMENT (cross-modal if different modalities)
    Detects: non-overlapping findings = INDEPENDENT

  async challenge_loop(contradiction: FindingComparison,
      agent: ForensicAgent, context_from_other: AgentFinding) -> ChallengeResult
    Sends challenge call via InterAgentBus with context_from_other as payload
    Agent re-examines and returns revised finding or maintains position
    If resolved: ChallengeResult(resolved=True)
    If not: ChallengeResult(resolved=False) → triggers Tribunal

  async trigger_tribunal(case: TribunalCase) -> None
    Logs HITL_CHECKPOINT with reason=TRIBUNAL_ESCALATION
    Formats Tribunal interface: both full reasoning chains, point of contradiction, evidence refs
    Awaits human judgment
    On judgment: logs TRIBUNAL_JUDGMENT to CustodyLogger

  async sign_report(report: ForensicReport) -> ForensicReport
    Computes SHA-256 of report JSON (excluding signature fields)
    Signs with Arbiter key
    Embeds signature and hash

Create reports/report_renderer.py:
  render_text(report: ForensicReport) -> str
    Renders ForensicReport as structured plain text / markdown

Tests in tests/test_agents/test_arbiter.py:
- test_cross_agent_comparison_detects_agreement
- test_cross_agent_comparison_detects_contradiction
- test_cross_agent_comparison_detects_cross_modal_confirmation
- test_challenge_loop_resolves_contradiction_with_context
- test_challenge_loop_unresolved_triggers_tribunal
- test_tribunal_logs_hitl_checkpoint
- test_report_contains_all_required_sections
- test_report_executive_summary_is_plain_language
- test_report_sign_embeds_signature_and_hash
- test_report_signature_verifiable_with_arbiter_public_key
- test_contested_finding_never_silently_resolved
- test_cross_modal_confirmed_finding_elevated_confidence
```

---

### 11.2 — Test Gate

```bash
uv run pytest tests/test_agents/test_arbiter.py -v

# Verify report has all required sections
uv run python -c "
from agents.arbiter import ForensicReport
import inspect
fields = ForensicReport.model_fields.keys()
required = ['executive_summary','per_agent_findings','cross_modal_confirmed',
            'contested_findings','tribunal_resolved','incomplete_findings',
            'chain_of_custody_log','uncertainty_statement','cryptographic_signature']
missing = [f for f in required if f not in fields]
print('Missing sections:', missing or 'NONE — all present')
"

uv run pytest tests/ -v --tb=short
```

**Expected:** No missing sections. All 12 Arbiter tests pass. Full suite green.

---

### 11.3 — Stage Completion Report Prompt

```
Stage 11 complete. Report:

BUILT: files
TESTS: all test names and status
REPORT SECTIONS: Confirm all required sections present — list them
CHALLENGE LOOP: Describe what the resolved contradiction test demonstrated
TRIBUNAL: Confirm HITL_CHECKPOINT with TRIBUNAL_ESCALATION reason was logged
SIGNATURE: Confirm report signature verifiable with arbiter public key
FULL SUITE: X tests, 0 failing
WHAT REMAINS: Stage 12 — End-to-End Integration
```

---

## Stage 12 — End-to-End Integration Test

### 12.1 — Kilo Code Prompt

```
Build the full end-to-end orchestration pipeline and integration test suite.
This stage wires all components together and validates the complete system.

Create:

orchestration/pipeline.py

ForensicCouncilPipeline: async class
  Constructor: loads all config, instantiates all agents, Arbiter, bus, memory systems

  async run_investigation(evidence_file_path: str, case_id: str,
      investigator_id: str) -> ForensicReport
    1. Ingest evidence → EvidenceArtifact (with hash)
    2. Create session_id UUID
    3. Instantiate all 5 agents with shared evidence artifact and session
    4. Run agents concurrently (asyncio.gather) — each runs full ReAct loop
    5. Collect all AgentLoopResults
    6. Pass to CouncilArbiter.deliberate()
    7. Return signed ForensicReport

  async handle_hitl_decision(session_id, checkpoint_id, decision: HumanDecision) -> None
    Routes human decision to correct agent's loop engine

Create:

orchestration/session_manager.py
  Tracks active sessions, active agent loops, pending HITL checkpoints
  Provides: get_active_checkpoints(session_id) -> list[HITLCheckpointState]
  Provides: get_investigator_brief(session_id, agent_id) -> str

tests/test_integration/test_e2e.py

Integration test — uses a real test image with known splice and EXIF anomalies
(create in conftest.py using Pillow + piexif: a JPEG with stripped software field
and a copy-paste splice in upper-left quadrant)

- test_full_pipeline_produces_signed_report
  - Runs complete pipeline on test image
  - Asserts report is not None
  - Asserts report.cryptographic_signature is not None
  - Asserts report.report_hash matches SHA-256 of report content

- test_agent1_detects_splice_in_test_image
  - Asserts at least one CONFIRMED finding from Agent 1 on upper-left region

- test_agent5_detects_absent_exif_field
  - Asserts Agent 5 flags absent software field as METADATA_STRIPPING_SUSPECTED

- test_chain_of_custody_log_not_empty
  - Asserts session chain has > 0 entries

- test_chain_of_custody_log_passes_verification
  - Calls custody_logger.verify_chain(session_id)
  - Asserts ChainVerificationReport.valid == True and broken_at is None

- test_evidence_version_tree_has_derivative_artifacts
  - Asserts version tree has at least 2 nodes (root + at least 1 derivative)

- test_report_uncertainty_statement_present
  - Asserts report.uncertainty_statement is not empty string

- test_all_confirmed_findings_are_calibrated
  - For every CONFIRMED finding in report: asserts calibrated == True

- test_no_finding_silently_merged_contested
  - Asserts len(report.contested_findings) >= 0 (contested list exists and is explicit)
```

---

### 12.2 — Test Gate

```bash
# Run integration tests (may take 2-5 minutes — real inference)
uv run pytest tests/test_integration/test_e2e.py -v -s

# Print full report to console
uv run python -c "
import asyncio
from orchestration.pipeline import ForensicCouncilPipeline
from reports.report_renderer import render_text

async def main():
    pipeline = ForensicCouncilPipeline()
    report = await pipeline.run_investigation(
        'tests/fixtures/test_spliced_image.jpg',
        case_id='TEST-001',
        investigator_id='TESTER'
    )
    print(render_text(report))

asyncio.run(main())
"

# Run full test suite — final check
uv run pytest tests/ -v --tb=short --cov=. --cov-report=term-missing
```

**Expected:** All 9 integration tests pass. Report renders with all sections. Overall coverage ≥ 75%.

---

### 12.3 — Final Stage Completion Report Prompt

```
Stage 12 — FINAL STAGE — complete. Provide full system completion report:

SYSTEM STATUS: What percentage of the architecture document has been implemented?

BUILT (ALL STAGES):
- Complete file listing with one-line descriptions

ALL TESTS PASSING:
- Total test count across all test files
- Any skipped or xfailed tests and reason

ARCHITECTURE COMPLIANCE CHECK:
For each of the 11 Foundational Principles, state: IMPLEMENTED / PARTIAL / NOT IMPLEMENTED
  1. ReAct Loop Architecture
  2. Dual-Layer Memory
  3. Human-in-the-Loop
  4. Task Decomposition Memory
  5. Agent Self-Reflection
  6. Confidence Calibration
  7. Evidence Versioning
  8. Adversarial Robustness Checks
  9. Audit Trail Signing
  10. Graceful Degradation
  11. Agent Disagreement Tribunal

WHAT IS NOT YET PRODUCTION-READY:
- List any stubs, mocks, or simplified implementations that need real replacements before court use
- List any external dependencies (models, APIs) that need production credentials

RECOMMENDED NEXT STEPS:
- 3-5 concrete actions to move from prototype to production hardening
```

---

## Quick Reference: Stage Dependency Map

```
Stage 0 (Infra)
    │
Stage 1 (Signing + Custody Logger)
    │
Stage 2 (Evidence Versioning)
    │
Stage 3 (Working + Episodic Memory)
    │
Stage 4 (ReAct Loop Engine + HITL)
    │
Stage 5 (Agent Base Class + Self-Reflection)
    ├── Stage 6 (Image + Metadata Tools)
    └── Stage 7 (Audio + Video Tools)
             │
         Stage 8 (Inter-Agent Communication)
             │
         Stage 9 (Confidence Calibration)
             │
         Stage 10 (Adversarial Robustness)
             │
         Stage 11 (Arbiter + Report)
             │
         Stage 12 (End-to-End Integration)
```

Each stage is a git commit checkpoint. Run `git commit -m "Stage N complete — X tests passing"` at every `Stage Completion Report` step.

---

*Implementation Roadmap v1.0 · Forensic Council · Architecture-Phase Aligned*
*No code in this document — all code generated by Kilo Code from prompts above.*
