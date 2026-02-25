"""
ReAct Loop Engine and HITL Checkpoint System for Forensic Council.

Implements the core THOUGHT → ACTION → OBSERVATION reasoning loop
with Human-in-the-Loop (HITL) checkpoints for forensic analysis.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Literal

from pydantic import BaseModel, Field

from core.custody_logger import CustodyLogger, EntryType
from core.tool_registry import ToolRegistry, ToolResult
from core.working_memory import WorkingMemory, WorkingMemoryState


class ReActStepType(str, Enum):
    """Types of steps in a ReAct loop."""
    THOUGHT = "THOUGHT"
    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"


class ReActStep(BaseModel):
    """A single step in the ReAct reasoning chain."""
    
    step_type: Literal["THOUGHT", "ACTION", "OBSERVATION"] = Field(
        ..., description="Type of reasoning step"
    )
    content: str = Field(..., description="The content of the step")
    tool_name: str | None = Field(
        default=None, description="Tool name if ACTION step"
    )
    tool_input: dict[str, Any] | None = Field(
        default=None, description="Tool input if ACTION step"
    )
    tool_output: dict[str, Any] | None = Field(
        default=None, description="Tool output if OBSERVATION step"
    )
    iteration: int = Field(..., description="Current iteration number")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the step"
    )


class HITLCheckpointReason(str, Enum):
    """Reasons for triggering a Human-in-the-Loop checkpoint."""
    ITERATION_CEILING_50PCT = "ITERATION_CEILING_50PCT"
    CONTESTED_FINDING = "CONTESTED_FINDING"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    SEVERITY_THRESHOLD_BREACH = "SEVERITY_THRESHOLD_BREACH"
    TRIBUNAL_ESCALATION = "TRIBUNAL_ESCALATION"


class HITLCheckpointStatus(str, Enum):
    """Status of a HITL checkpoint."""
    PAUSED = "PAUSED"
    RESUMED = "RESUMED"
    OVERRIDDEN = "OVERRIDDEN"
    TERMINATED = "TERMINATED"


class HITLCheckpointState(BaseModel):
    """State of a Human-in-the-Loop checkpoint."""
    
    checkpoint_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique checkpoint identifier"
    )
    agent_id: str = Field(..., description="Agent that triggered checkpoint")
    session_id: uuid.UUID = Field(..., description="Session ID")
    reason: HITLCheckpointReason = Field(
        ..., description="Why checkpoint was triggered"
    )
    current_finding_summary: str = Field(
        default="", description="Summary of findings so far"
    )
    paused_at_iteration: int = Field(
        ..., description="Iteration at which loop was paused"
    )
    investigator_brief: str = Field(
        default="", description="Brief for the human investigator"
    )
    status: Literal["PAUSED", "RESUMED", "OVERRIDDEN", "TERMINATED"] = Field(
        default="PAUSED", description="Current checkpoint status"
    )
    serialized_state: dict[str, Any] | None = Field(
        default=None, description="Serialized working memory state"
    )


class HumanDecisionType(str, Enum):
    """Types of human decisions in HITL."""
    APPROVE = "APPROVE"
    REDIRECT = "REDIRECT"
    OVERRIDE = "OVERRIDE"
    TERMINATE = "TERMINATE"
    ESCALATE = "ESCALATE"


class HumanDecision(BaseModel):
    """A human decision in response to a HITL checkpoint."""
    
    decision_type: Literal["APPROVE", "REDIRECT", "OVERRIDE", "TERMINATE", "ESCALATE"] = Field(
        ..., description="Type of decision made"
    )
    investigator_id: str = Field(..., description="ID of the human investigator")
    notes: str = Field(default="", description="Notes from the investigator")
    override_finding: dict[str, Any] | None = Field(
        default=None, description="Override finding if OVERRIDE decision"
    )
    redirect_context: str | None = Field(
        default=None, description="New context/direction if REDIRECT decision"
    )


class AgentFindingStatus(str, Enum):
    """Status of an agent finding."""
    CONFIRMED = "CONFIRMED"
    CONTESTED = "CONTESTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    INCOMPLETE = "INCOMPLETE"


class AgentFinding(BaseModel):
    """A finding produced by an agent."""
    
    finding_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique finding identifier"
    )
    agent_id: str = Field(..., description="Agent that produced the finding")
    finding_type: str = Field(..., description="Type of finding")
    confidence_raw: float = Field(
        ..., ge=0.0, le=1.0,
        description="Raw confidence score (0-1)"
    )
    calibrated: bool = Field(
        default=False, description="Whether confidence has been calibrated"
    )
    status: Literal["CONFIRMED", "CONTESTED", "INCONCLUSIVE", "INCOMPLETE"] = Field(
        default="CONFIRMED", description="Finding status"
    )
    robustness_caveat: bool = Field(
        default=False, description="Whether finding has robustness caveat"
    )
    robustness_caveat_detail: str | None = Field(
        default=None, description="Detail about robustness caveat"
    )
    evidence_refs: list[uuid.UUID] = Field(
        default_factory=list, description="References to evidence artifacts"
    )
    reasoning_summary: str = Field(
        default="", description="Summary of reasoning that led to finding"
    )


class ReActLoopResult(BaseModel):
    """Result of a completed ReAct loop."""
    
    session_id: uuid.UUID = Field(..., description="Session ID")
    agent_id: str = Field(..., description="Agent ID")
    completed: bool = Field(
        default=False, description="Whether loop completed normally"
    )
    terminated_by_human: bool = Field(
        default=False, description="Whether loop was terminated by human"
    )
    findings: list[AgentFinding] = Field(
        default_factory=list, description="Findings produced"
    )
    hitl_checkpoints: list[HITLCheckpointState] = Field(
        default_factory=list, description="HITL checkpoints encountered"
    )
    total_iterations: int = Field(default=0, description="Total iterations run")
    react_chain: list[ReActStep] = Field(
        default_factory=list, description="Full reasoning chain"
    )


# Type for LLM step generators - async function that returns next ReActStep
LLMStepGenerator = Callable[
    [list[ReActStep], WorkingMemoryState],
    Coroutine[Any, Any, ReActStep | None]
]


class ReActLoopEngine:
    """
    Core ReAct (Reasoning + Acting) loop engine.
    
    Implements the THOUGHT → ACTION → OBSERVATION cycle with:
    - Human-in-the-Loop checkpoints at trigger conditions
    - Graceful degradation on tool unavailability
    - Full audit logging to chain of custody
    """
    
    def __init__(
        self,
        agent_id: str,
        session_id: uuid.UUID,
        iteration_ceiling: int,
        working_memory: WorkingMemory,
        custody_logger: CustodyLogger,
        redis_client: Any = None,  # Redis client for HITL checkpoint storage
        hitl_timeout: float = 3600.0  # Timeout for HITL resume wait (1 hour default)
    ) -> None:
        """
        Initialize the ReAct loop engine.
        
        Args:
            agent_id: ID of the agent running this loop
            session_id: Session ID for this analysis
            iteration_ceiling: Maximum iterations before forced stop
            working_memory: Working memory for task tracking
            custody_logger: Logger for chain of custody
            redis_client: Redis client for HITL checkpoint storage
            hitl_timeout: Timeout in seconds for waiting on HITL resume
        """
        self.agent_id = agent_id
        self.session_id = session_id
        self.iteration_ceiling = iteration_ceiling
        self.working_memory = working_memory
        self.custody_logger = custody_logger
        self.redis_client = redis_client
        self.hitl_timeout = hitl_timeout
        
        # Internal state
        self._current_iteration = 0
        self._react_chain: list[ReActStep] = []
        self._findings: list[AgentFinding] = []
        self._hitl_checkpoints: list[HITLCheckpointState] = []
        self._terminated = False
        self._current_checkpoint: HITLCheckpointState | None = None
        self._resume_event: asyncio.Event | None = None
        self._pending_decision: HumanDecision | None = None

    async def run(
        self,
        initial_thought: str,
        tool_registry: ToolRegistry,
        llm_generator: LLMStepGenerator | None = None
    ) -> ReActLoopResult:
        """
        Run the ReAct loop from an initial thought.
        
        Args:
            initial_thought: The starting thought for the loop
            tool_registry: Registry of available tools
            llm_generator: Async function that generates next step from LLM.
                          If None, uses a simple mock that signals completion.
                          
        Returns:
            ReActLoopResult with findings and reasoning chain
        """
        # Initialize working memory state
        state = await self.working_memory.get_state(self.session_id)
        if state is None:
            state = await self.working_memory.create_state(
                session_id=self.session_id,
                agent_id=self.agent_id
            )

        # Create initial THOUGHT step
        initial_step = ReActStep(
            step_type="THOUGHT",
            content=initial_thought,
            iteration=0
        )
        self._react_chain.append(initial_step)
        await self._log_step(initial_step)
        
        self._current_iteration = 0

        # Main loop
        while not self._terminated and self._current_iteration < self.iteration_ceiling:
            # Get current state
            state = await self.working_memory.get_state(self.session_id)
            if state is None:
                break

            # Check HITL triggers before proceeding
            hitl_reason = await self.check_hitl_triggers(state)
            if hitl_reason is not None:
                checkpoint = await self.pause_for_hitl(
                    reason=hitl_reason,
                    brief=f"Paused at iteration {self._current_iteration} due to {hitl_reason.value}"
                )
                self._hitl_checkpoints.append(checkpoint)
                
                # Wait for resume signal (in real implementation, this would be external)
                # For now, we check if a decision was set via resume_from_hitl
                if self._resume_event is None:
                    self._resume_event = asyncio.Event()
                
                # In test mode, we might have a pending decision already
                if self._pending_decision is None:
                    # Wait for external resume (with timeout for safety)
                    try:
                        await asyncio.wait_for(
                            self._resume_event.wait(),
                            timeout=self.hitl_timeout
                        )
                    except asyncio.TimeoutError:
                        # Timeout - terminate loop
                        self._terminated = True
                        break
                
                # Process the decision
                if self._pending_decision is not None:
                    await self.resume_from_hitl(
                        checkpoint.checkpoint_id,
                        self._pending_decision
                    )
                    self._pending_decision = None
                    self._resume_event.clear()
                
                # Check if terminated after HITL
                if self._terminated:
                    break

            # Increment iteration
            self._current_iteration += 1

            # Get next step from LLM
            if llm_generator is not None:
                next_step = await llm_generator(self._react_chain, state)
            else:
                # Default: signal completion after initial thought
                next_step = None

            if next_step is None:
                # LLM signals completion
                break

            next_step.iteration = self._current_iteration
            self._react_chain.append(next_step)
            await self._log_step(next_step)

            # Handle ACTION steps
            if next_step.step_type == "ACTION" and next_step.tool_name:
                tool_result = await tool_registry.call(
                    tool_name=next_step.tool_name,
                    input_data=next_step.tool_input or {},
                    agent_id=self.agent_id,
                    session_id=self.session_id,
                    custody_logger=self.custody_logger
                )

                # Create OBSERVATION step
                observation = ReActStep(
                    step_type="OBSERVATION",
                    content=self._format_tool_result(tool_result),
                    tool_name=next_step.tool_name,
                    tool_output=tool_result.model_dump(),
                    iteration=self._current_iteration
                )
                self._react_chain.append(observation)
                await self._log_step(observation)

                # Check for tool unavailability HITL trigger
                if tool_result.unavailable:
                    hitl_reason = await self.check_hitl_triggers(state)
                    if hitl_reason == HITLCheckpointReason.TOOL_UNAVAILABLE:
                        checkpoint = await self.pause_for_hitl(
                            reason=hitl_reason,
                            brief=f"Tool '{next_step.tool_name}' unavailable"
                        )
                        self._hitl_checkpoints.append(checkpoint)

            # Update working memory with current iteration
            await self.working_memory.update_state(
                session_id=self.session_id,
                updates={"current_iteration": self._current_iteration}
            )

        # Build result
        return ReActLoopResult(
            session_id=self.session_id,
            agent_id=self.agent_id,
            completed=(self._current_iteration >= self.iteration_ceiling or 
                      not self._terminated),
            terminated_by_human=self._terminated,
            findings=self._findings,
            hitl_checkpoints=self._hitl_checkpoints,
            total_iterations=self._current_iteration,
            react_chain=self._react_chain
        )

    async def check_hitl_triggers(
        self, 
        state: WorkingMemoryState
    ) -> HITLCheckpointReason | None:
        """
        Check if any HITL trigger conditions are met.
        
        Args:
            state: Current working memory state
            
        Returns:
            HITLCheckpointReason if triggered, None otherwise
        """
        # Trigger at 50% of iteration ceiling without COMPLETE task
        half_ceiling = self.iteration_ceiling // 2
        if self._current_iteration >= half_ceiling:
            # Check if there's a COMPLETE task
            has_complete = any(
                task.status == "COMPLETE" for task in state.tasks
            )
            if not has_complete and self._current_iteration == half_ceiling:
                return HITLCheckpointReason.ITERATION_CEILING_50PCT

        # Check for contested findings
        for task in state.tasks:
            if task.status == "CONTESTED":
                return HITLCheckpointReason.CONTESTED_FINDING

        # Check for severity threshold breach (if findings have high severity)
        # This would be checked against findings in a real implementation
        # For now, we check if any task has severity_threshold flag
        for task in state.tasks:
            if hasattr(task, 'severity_threshold') and task.severity_threshold:
                return HITLCheckpointReason.SEVERITY_THRESHOLD_BREACH

        return None

    async def pause_for_hitl(
        self,
        reason: HITLCheckpointReason,
        brief: str
    ) -> HITLCheckpointState:
        """
        Pause the loop for Human-in-the-Loop intervention.
        
        Args:
            reason: Why the checkpoint was triggered
            brief: Brief description for the investigator
            
        Returns:
            HITLCheckpointState with PAUSED status
        """
        # Serialize working memory state
        state = await self.working_memory.get_state(self.session_id)
        serialized_state = state.model_dump() if state else {}

        # Create checkpoint
        checkpoint = HITLCheckpointState(
            agent_id=self.agent_id,
            session_id=self.session_id,
            reason=reason,
            paused_at_iteration=self._current_iteration,
            investigator_brief=brief,
            status="PAUSED",
            serialized_state=serialized_state
        )

        # Log HITL checkpoint
        await self.custody_logger.log_entry(
            agent_id=self.agent_id,
            session_id=self.session_id,
            entry_type=EntryType.HITL_CHECKPOINT,
            content={
                "checkpoint_id": str(checkpoint.checkpoint_id),
                "reason": reason.value,
                "paused_at_iteration": self._current_iteration,
                "brief": brief
            }
        )

        # Store checkpoint in Redis
        if self.redis_client is not None:
            key = f"hitl:{self.session_id}:{self.agent_id}"
            await self.redis_client.set(
                key,
                json.dumps(checkpoint.model_dump(), default=str)
            )

        self._current_checkpoint = checkpoint
        return checkpoint

    async def resume_from_hitl(
        self,
        checkpoint_id: uuid.UUID,
        decision: HumanDecision
    ) -> None:
        """
        Resume the loop after a HITL decision.
        
        Args:
            checkpoint_id: ID of the checkpoint to resume from
            decision: The human decision
        """
        # Log human intervention
        await self.custody_logger.log_entry(
            agent_id=self.agent_id,
            session_id=self.session_id,
            entry_type=EntryType.HUMAN_INTERVENTION,
            content={
                "checkpoint_id": str(checkpoint_id),
                "decision_type": decision.decision_type,
                "investigator_id": decision.investigator_id,
                "notes": decision.notes
            }
        )

        # Handle different decision types
        if decision.decision_type == "TERMINATE":
            self._terminated = True
            if self._current_checkpoint:
                self._current_checkpoint.status = "TERMINATED"
            return

        if decision.decision_type == "OVERRIDE" and decision.override_finding:
            # Create a finding from the override
            finding = AgentFinding(
                agent_id=self.agent_id,
                finding_type="HUMAN_OVERRIDE",
                confidence_raw=1.0,  # Human judgment is certain
                status="CONFIRMED",
                reasoning_summary=decision.notes
            )
            self._findings.append(finding)
            if self._current_checkpoint:
                self._current_checkpoint.status = "OVERRIDDEN"

        if decision.decision_type == "REDIRECT" and decision.redirect_context:
            # Inject redirect context into working memory
            await self.working_memory.update_state(
                session_id=self.session_id,
                updates={"redirect_context": decision.redirect_context}
            )
            if self._current_checkpoint:
                self._current_checkpoint.status = "RESUMED"

        if decision.decision_type == "APPROVE":
            if self._current_checkpoint:
                self._current_checkpoint.status = "RESUMED"

        if decision.decision_type == "ESCALATE":
            # Mark for tribunal escalation
            await self.working_memory.update_state(
                session_id=self.session_id,
                updates={"tribunal_escalation": True}
            )
            if self._current_checkpoint:
                self._current_checkpoint.status = "RESUMED"

        # Clear checkpoint from Redis
        if self.redis_client is not None:
            key = f"hitl:{self.session_id}:{self.agent_id}"
            await self.redis_client.delete(key)

        # Signal resume
        self._pending_decision = decision
        if self._resume_event:
            self._resume_event.set()

    async def _log_step(self, step: ReActStep) -> None:
        """Log a ReAct step to the custody logger."""
        # Map step type to EntryType
        step_type_to_entry_type = {
            "THOUGHT": EntryType.THOUGHT,
            "ACTION": EntryType.ACTION,
            "OBSERVATION": EntryType.OBSERVATION,
        }
        entry_type = step_type_to_entry_type.get(
            step.step_type, EntryType.THOUGHT
        )
        await self.custody_logger.log_entry(
            agent_id=self.agent_id,
            session_id=self.session_id,
            entry_type=entry_type,
            content={
                "step_type": step.step_type,
                "content": step.content,
                "iteration": step.iteration,
                "tool_name": step.tool_name,
                "tool_input": step.tool_input,
                "timestamp": step.timestamp_utc.isoformat()
            }
        )

    def _format_tool_result(self, result: ToolResult) -> str:
        """Format a tool result for observation content."""
        if result.unavailable:
            return f"Tool '{result.tool_name}' is unavailable. Error: {result.error}"
        if result.success:
            return f"Tool '{result.tool_name}' succeeded. Output: {result.output}"
        return f"Tool '{result.tool_name}' failed. Error: {result.error}"

    def add_finding(self, finding: AgentFinding) -> None:
        """Add a finding to the result."""
        self._findings.append(finding)

    def set_pending_decision(self, decision: HumanDecision) -> None:
        """Set a pending decision for HITL resume (used in tests)."""
        self._pending_decision = decision
        if self._resume_event:
            self._resume_event.set()
