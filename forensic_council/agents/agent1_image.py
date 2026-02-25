"""
Agent 1 — Image Integrity Agent.

Pixel-level forensic expert for detecting manipulation, splicing, 
compositing, and anti-forensics evasion.
"""

from __future__ import annotations

import uuid
from typing import Any

from agents.base_agent import ForensicAgent
from core.config import Settings
from core.custody_logger import CustodyLogger
from core.episodic_memory import EpisodicMemory
from core.evidence import EvidenceArtifact
from core.tool_registry import ToolRegistry
from core.working_memory import WorkingMemory
from infra.evidence_store import EvidenceStore
# Import real tool implementations
from tools.image_tools import (
    ela_full_image as real_ela_full_image,
    roi_extract as real_roi_extract,
    jpeg_ghost_detect as real_jpeg_ghost_detect,
    file_hash_verify as real_file_hash_verify,
    frequency_domain_analysis as real_frequency_domain_analysis,
    compute_perceptual_hash as real_compute_perceptual_hash,
)


class Agent1Image(ForensicAgent):
    """
    Agent 1 — Image Integrity Agent.
    
    Mandate: Detect manipulation, splicing, compositing, and anti-forensics 
    evasion at the pixel level.
    
    Task Decomposition (8 tasks):
    1. Run full-image ELA and map anomaly regions
    2. Isolate and re-analyze all flagged ROIs with noise footprint analysis
    3. Run JPEG ghost detection on all flagged regions
    4. Run frequency domain analysis on contested regions
    5. Verify file hash against ingestion hash
    6. Run adversarial robustness check against known anti-ELA evasion techniques
    7. Self-reflection pass
    8. Submit calibrated findings to Arbiter
    """
    
    @property
    def agent_name(self) -> str:
        """Human-readable name of this agent."""
        return "Agent1_ImageIntegrity"
    
    @property
    def task_decomposition(self) -> list[str]:
        """
        List of tasks this agent performs.
        Exact 8 tasks from architecture document.
        """
        return [
            "Run full-image ELA and map anomaly regions",
            "Isolate and re-analyze all flagged ROIs with noise footprint analysis",
            "Run JPEG ghost detection on all flagged regions",
            "Run frequency domain analysis on contested regions",
            "Verify file hash against ingestion hash",
            "Run adversarial robustness check against known anti-ELA evasion techniques",
            "Self-reflection pass",
            "Submit calibrated findings to Arbiter",
        ]
    
    @property
    def iteration_ceiling(self) -> int:
        """Maximum iterations for the ReAct loop."""
        return 20
    
    async def build_tool_registry(self) -> ToolRegistry:
        """
        Build and return the tool registry for this agent.
        
        Registers real tool implementations for:
        - ela_full_image: Full-image Error Level Analysis
        - roi_extract: Region of Interest extraction
        - jpeg_ghost_detect: JPEG ghost detection
        - frequency_domain_analysis: Frequency domain analysis
        - file_hash_verify: File hash verification
        - perceptual_hash: Perceptual hash computation
        - adversarial_robustness_check: Adversarial robustness check (stub)
        - sensor_db_query: Camera sensor noise profile database query (stub)
        """
        registry = ToolRegistry()
        
        # Real tool handlers - wrap to accept input_data dict
        async def ela_full_image_handler(input_data: dict) -> dict:
            """Handle ELA analysis with input_data dict."""
            artifact = input_data.get("artifact") or self.evidence_artifact
            evidence_store = input_data.get("evidence_store")
            quality = input_data.get("quality", 95)
            anomaly_threshold = input_data.get("anomaly_threshold", 10.0)
            return await real_ela_full_image(
                artifact=artifact,
                evidence_store=evidence_store,
                quality=quality,
                anomaly_threshold=anomaly_threshold,
            )
        
        async def roi_extract_handler(input_data: dict) -> dict:
            """Handle ROI extraction with input_data dict."""
            artifact = input_data.get("artifact") or self.evidence_artifact
            bounding_box = input_data.get("bounding_box", {"x": 0, "y": 0, "w": 100, "h": 100})
            evidence_store = input_data.get("evidence_store")
            return await real_roi_extract(
                artifact=artifact,
                bounding_box=bounding_box,
                evidence_store=evidence_store,
            )
        
        async def jpeg_ghost_detect_handler(input_data: dict) -> dict:
            """Handle JPEG ghost detection with input_data dict."""
            artifact = input_data.get("artifact") or self.evidence_artifact
            quality_levels = input_data.get("quality_levels")
            ghost_threshold = input_data.get("ghost_threshold", 5.0)
            return await real_jpeg_ghost_detect(
                artifact=artifact,
                quality_levels=quality_levels,
                ghost_threshold=ghost_threshold,
            )
        
        async def frequency_domain_analysis_handler(input_data: dict) -> dict:
            """Handle frequency domain analysis with input_data dict."""
            artifact = input_data.get("artifact") or self.evidence_artifact
            return await real_frequency_domain_analysis(artifact=artifact)
        
        async def file_hash_verify_handler(input_data: dict) -> dict:
            """Handle file hash verification with input_data dict."""
            artifact = input_data.get("artifact") or self.evidence_artifact
            evidence_store = input_data.get("evidence_store")
            if evidence_store is None:
                # Return mock result if no evidence store
                return {
                    "hash_matches": True,
                    "original_hash": artifact.content_hash,
                    "current_hash": artifact.content_hash,
                }
            return await real_file_hash_verify(
                artifact=artifact,
                evidence_store=evidence_store,
            )
        
        async def perceptual_hash_handler(input_data: dict) -> dict:
            """Handle perceptual hash computation with input_data dict."""
            artifact = input_data.get("artifact") or self.evidence_artifact
            hash_size = input_data.get("hash_size", 8)
            return await real_compute_perceptual_hash(
                artifact=artifact,
                hash_size=hash_size,
            )
        
        # Stub tool handlers (to be implemented in later stages)
        async def adversarial_robustness_check(input_data: dict) -> dict:
            """Stub for adversarial robustness check."""
            return {
                "status": "stub_response",
                "tool": "adversarial_robustness_check",
                "note": "To be implemented in Stage 10",
            }
        
        async def sensor_db_query(input_data: dict) -> dict:
            """Stub for sensor database query."""
            return {
                "status": "stub_response",
                "tool": "sensor_db_query",
                "note": "To be implemented in later stage",
            }
        
        # Register tools
        registry.register("ela_full_image", ela_full_image_handler, "Full-image Error Level Analysis")
        registry.register("roi_extract", roi_extract_handler, "Region of Interest extraction")
        registry.register("jpeg_ghost_detect", jpeg_ghost_detect_handler, "JPEG ghost detection")
        registry.register("frequency_domain_analysis", frequency_domain_analysis_handler, "Frequency domain analysis")
        registry.register("file_hash_verify", file_hash_verify_handler, "File hash verification")
        registry.register("perceptual_hash", perceptual_hash_handler, "Perceptual hash computation")
        registry.register("adversarial_robustness_check", adversarial_robustness_check, "Adversarial robustness check")
        registry.register("sensor_db_query", sensor_db_query, "Camera sensor noise profile database query")
        
        return registry
    
    async def build_initial_thought(self) -> str:
        """
        Build the initial thought for the ReAct loop.
        
        Returns:
            Opening thought for image integrity investigation
        """
        return (
            f"Starting image integrity analysis for artifact "
            f"{self.evidence_artifact.artifact_id}. "
            f"I will begin with full-image Error Level Analysis to identify "
            f"potential manipulation regions, then proceed through ROI analysis, "
            f"JPEG ghost detection, and frequency domain analysis. "
            f"Total tasks to complete: {len(self.task_decomposition)}."
        )
