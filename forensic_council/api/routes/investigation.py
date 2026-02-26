"""
Investigation Routes
====================

Routes for starting and managing forensic investigations.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse

from api.schemas import (
    AgentFindingDTO,
    BriefUpdate,
    InvestigationRequest,
    InvestigationResponse,
    ReportDTO,
)
from core.logging import get_logger
from orchestration.pipeline import ForensicCouncilPipeline
from orchestration.session_manager import SessionManager

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["investigation"])

# Store active pipelines and their WebSocket connections
_active_pipelines: dict[str, ForensicCouncilPipeline] = {}
_websocket_connections: dict[str, list] = {}

# Allowed MIME types
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "video/mp4",
    "audio/wav",
    "audio/mpeg",
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


async def broadcast_update(session_id: str, update: BriefUpdate):
    """Broadcast a WebSocket update to all connected clients."""
    if session_id in _websocket_connections:
        for ws in _websocket_connections[session_id]:
            try:
                await ws.send_json(update.model_dump())
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")


async def run_investigation_task(
    session_id: str,
    evidence_file_path: str,
    case_id: str,
    investigator_id: str,
):
    """Background task to run the investigation."""
    pipeline = ForensicCouncilPipeline()
    _active_pipelines[session_id] = pipeline

    try:
        # Send initial update
        await broadcast_update(
            session_id,
            BriefUpdate(
                type="AGENT_UPDATE",
                session_id=session_id,
                message="Starting investigation...",
                data={"status": "starting"},
            )
        )

        # Run the investigation
        report = await pipeline.run_investigation(
            evidence_file_path=evidence_file_path,
            case_id=case_id,
            investigator_id=investigator_id,
        )

        # Send completion update
        await broadcast_update(
            session_id,
            BriefUpdate(
                type="PIPELINE_COMPLETE",
                session_id=session_id,
                message="Investigation complete",
                data={"report_id": report.report_id},
            )
        )

        # Store the report for retrieval
        pipeline._final_report = report

    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        await broadcast_update(
            session_id,
            BriefUpdate(
                type="ERROR",
                session_id=session_id,
                message=f"Investigation failed: {str(e)}",
                data={"error": str(e)},
            )
        )
    finally:
        # Close WebSocket connections
        if session_id in _websocket_connections:
            for ws in _websocket_connections[session_id]:
                try:
                    await ws.close()
                except Exception:
                    pass
            _websocket_connections[session_id] = []


@router.post("/investigate", response_model=InvestigationResponse)
async def investigate(
    file: UploadFile = File(...),
    case_id: str = Form(...),
    investigator_id: str = Form(...),
):
    """
    Start a forensic investigation on uploaded evidence.
    
    Accepts multipart/form-data with:
    - file: The evidence file (image, video, or audio)
    - case_id: Case identifier
    - investigator_id: ID of the investigator
    """
    # Validate file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )

    # Validate MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_MIME_TYPES)}"
        )

    # Create session ID
    session_id = str(uuid4())

    # Save uploaded file to temporary location asynchronously
    content = await file.read()
    
    def write_temp_file(data: bytes, ext: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(data)
            return tmp_file.name
            
    evidence_file_path = await asyncio.to_thread(write_temp_file, content, Path(file.filename).suffix)

    try:
        # Start investigation in background
        asyncio.create_task(
            run_investigation_task(
                session_id=session_id,
                evidence_file_path=evidence_file_path,
                case_id=case_id,
                investigator_id=investigator_id,
            )
        )

        return InvestigationResponse(
            session_id=session_id,
            case_id=case_id,
            status="started",
            message="Investigation started successfully"
        )

    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(evidence_file_path):
            os.remove(evidence_file_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/report")
async def get_report(session_id: str):
    """Get the report for a completed investigation."""
    if session_id not in _active_pipelines:
        # Check if we have a stored report
        raise HTTPException(status_code=404, detail="Session not found")

    pipeline = _active_pipelines[session_id]

    # Check if investigation is still running
    if not hasattr(pipeline, "_final_report"):
        return JSONResponse(
            status_code=202,
            content={"status": "in_progress", "message": "Investigation still running"}
        )

    report = pipeline._final_report

    # Convert to DTO
    per_agent_findings = {}
    for agent_id, findings in report.per_agent_findings.items():
        per_agent_findings[agent_id] = [
            AgentFindingDTO(
                finding_id=str(f.get("finding_id", "")),
                agent_id=agent_id,
                agent_name=f.get("agent_name", agent_id),
                finding_type=f.get("finding_type", "unknown"),
                status=f.get("status", "unknown"),
                confidence_raw=f.get("confidence_raw", 0.0),
                calibrated=f.get("calibrated", False),
                calibrated_probability=f.get("calibrated_probability"),
                court_statement=f.get("court_statement"),
                robustness_caveat=f.get("robustness_caveat", False),
                robustness_caveat_detail=f.get("robustness_caveat_detail"),
                reasoning_summary=f.get("reasoning_summary", ""),
            )
            for f in findings
        ]

    cross_modal_confirmed = [
        AgentFindingDTO(
            finding_id=str(f.get("finding_id", "")),
            agent_id=f.get("agent_id", ""),
            agent_name=f.get("agent_name", ""),
            finding_type=f.get("finding_type", "unknown"),
            status=f.get("status", "unknown"),
            confidence_raw=f.get("confidence_raw", 0.0),
            calibrated=f.get("calibrated", False),
            calibrated_probability=f.get("calibrated_probability"),
            court_statement=f.get("court_statement"),
            robustness_caveat=f.get("robustness_caveat", False),
            robustness_caveat_detail=f.get("robustness_caveat_detail"),
            reasoning_summary=f.get("reasoning_summary", ""),
        )
        for f in report.cross_modal_confirmed
    ]

    incomplete_findings = [
        AgentFindingDTO(
            finding_id=str(f.get("finding_id", "")),
            agent_id=f.get("agent_id", ""),
            agent_name=f.get("agent_name", ""),
            finding_type=f.get("finding_type", "unknown"),
            status=f.get("status", "unknown"),
            confidence_raw=f.get("confidence_raw", 0.0),
            calibrated=f.get("calibrated", False),
            calibrated_probability=f.get("calibrated_probability"),
            court_statement=f.get("court_statement"),
            robustness_caveat=f.get("robustness_caveat", False),
            robustness_caveat_detail=f.get("robustness_caveat_detail"),
            reasoning_summary=f.get("reasoning_summary", ""),
        )
        for f in report.incomplete_findings
    ]

    return ReportDTO(
        report_id=str(report.report_id),
        session_id=session_id,
        case_id=report.case_id,
        executive_summary=report.executive_summary,
        per_agent_findings=per_agent_findings,
        cross_modal_confirmed=cross_modal_confirmed,
        contested_findings=report.contested_findings,
        tribunal_resolved=report.tribunal_resolved,
        incomplete_findings=incomplete_findings,
        uncertainty_statement=report.uncertainty_statement,
        cryptographic_signature=report.cryptographic_signature,
        report_hash=report.report_hash,
        signed_utc=report.signed_utc.isoformat() if hasattr(report.signed_utc, 'isoformat') else str(report.signed_utc),
    )


@router.get("/sessions/{session_id}/brief/{agent_id}")
async def get_brief(session_id: str, agent_id: str):
    """Get the current investigator brief for an agent."""
    if session_id not in _active_pipelines:
        raise HTTPException(status_code=404, detail="Session not found")

    pipeline = _active_pipelines[session_id]

    # Get brief from working memory
    if pipeline.working_memory:
        brief = await pipeline.working_memory.get_agent_brief(session_id, agent_id)
        if brief:
            return {"brief": brief}

    return {"brief": "No brief available yet."}


@router.get("/sessions/{session_id}/checkpoints")
async def get_checkpoints(session_id: str):
    """Get pending HITL checkpoints for a session."""
    if session_id not in _active_pipelines:
        raise HTTPException(status_code=404, detail="Session not found")

    # Return empty list for now - HITL checkpoint tracking would need to be implemented
    return []


def register_websocket(session_id: str, websocket):
    """Register a WebSocket connection for a session."""
    if session_id not in _websocket_connections:
        _websocket_connections[session_id] = []
    _websocket_connections[session_id].append(websocket)


def unregister_websocket(session_id: str, websocket):
    """Unregister a WebSocket connection."""
    if session_id in _websocket_connections:
        if websocket in _websocket_connections[session_id]:
            _websocket_connections[session_id].remove(websocket)
