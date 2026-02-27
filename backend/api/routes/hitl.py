"""
HITL Routes
===========

Routes for human-in-the-loop decision handling.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import HITLDecisionRequest

router = APIRouter(prefix="/api/v1/hitl", tags=["hitl"])


@router.post("/decision")
async def submit_decision(decision: HITLDecisionRequest):
    """
    Submit a human-in-the-loop decision for a checkpoint.
    
    Args:
        decision: The decision including session_id, checkpoint_id, agent_id,
                  and the decision type (APPROVE, REDIRECT, OVERRIDE, TERMINATE, TRIBUNAL)
    """
    # TODO: Integrate with pipeline's HITL handling
    # For now, return a placeholder response
    
    # This would route the decision to the pipeline via the inter-agent bus
    # or working memory for the agent to act on
    
    return {
        "status": "received",
        "message": f"Decision {decision.decision} received for checkpoint {decision.checkpoint_id}",
        "session_id": decision.session_id,
    }
