"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { AGENTS_DATA } from "@/lib/constants";
import { AgentResult } from "@/types";
import { createLiveSocket, BriefUpdate, HITLCheckpoint } from "@/lib/api";

type SimulationStatus = "idle" | "analyzing" | "initiating" | "processing" | "complete";

type UseSimulationProps = {
    onAgentComplete?: (result: AgentResult) => void;
    onComplete?: () => void;
    playSound?: (type: "success" | "agent" | "complete" | "think") => void;
};

export const useSimulation = ({ onAgentComplete, onComplete, playSound }: UseSimulationProps) => {
    const [status, setStatus] = useState<SimulationStatus>("idle");
    const [completedAgents, setCompletedAgents] = useState<AgentResult[]>([]);
    const [currentAgentIndex, setCurrentAgentIndex] = useState(-1);
    const [currentThinkingPhrase, setCurrentThinkingPhrase] = useState("");
    const [hitlCheckpoint, setHitlCheckpoint] = useState<HITLCheckpoint | null>(null);
    const [sessionId, setSessionId] = useState<string | null>(null);
    
    const wsRef = useRef<WebSocket | null>(null);
    const thinkingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

    // Store callbacks in refs to avoid triggering effect on every render
    const onAgentCompleteRef = useRef(onAgentComplete);
    const onCompleteRef = useRef(onComplete);
    const playSoundRef = useRef(playSound);

    // Update refs when props change
    useEffect(() => {
        onAgentCompleteRef.current = onAgentComplete;
        onCompleteRef.current = onComplete;
        playSoundRef.current = playSound;
    }, [onAgentComplete, onComplete, playSound]);

    // Connect to WebSocket when sessionId changes
    useEffect(() => {
        if (!sessionId) {
            return;
        }

        // Disconnect existing WebSocket
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }

        const handleMessage = (update: BriefUpdate) => {
            console.log("[WebSocket] Received update:", update);

            switch (update.type) {
                case "AGENT_UPDATE":
                    // Update agent thinking/status
                    if (update.agent_id && update.data) {
                        const agentData = update.data as { status?: string; thinking?: string };
                        if (agentData.thinking) {
                            setCurrentThinkingPhrase(agentData.thinking);
                        }
                    }
                    setStatus("analyzing");
                    break;

                case "HITL_CHECKPOINT":
                    // Show HITL checkpoint
                    if (update.data) {
                        const checkpoint: HITLCheckpoint = {
                            checkpoint_id: update.data.checkpoint_id as string,
                            session_id: update.session_id,
                            agent_id: update.agent_id || "",
                            agent_name: update.agent_name || "",
                            brief_text: update.message,
                            decision_needed: "APPROVE, REDIRECT, or TERMINATE",
                            created_at: new Date().toISOString(),
                        };
                        setHitlCheckpoint(checkpoint);
                    }
                    break;

                case "AGENT_COMPLETE":
                    // Mark agent as complete
                    if (update.agent_id) {
                        const agent = AGENTS_DATA.find(a => a.id === update.agent_id);
                        if (agent) {
                            const result: AgentResult = {
                                id: agent.id,
                                name: agent.name,
                                role: agent.role,
                                result: update.message,
                                confidence: 0.85,
                                thinking: update.message,
                            };
                            setCompletedAgents(prev => [...prev, result]);
                            onAgentCompleteRef.current?.(result);
                            playSoundRef.current?.("agent");
                        }
                    }
                    break;

                case "PIPELINE_COMPLETE":
                    // All agents complete
                    setStatus("complete");
                    onCompleteRef.current?.();
                    playSoundRef.current?.("complete");
                    
                    // Close WebSocket
                    if (wsRef.current) {
                        wsRef.current.close();
                        wsRef.current = null;
                    }
                    break;

                case "ERROR":
                    console.error("[WebSocket] Error:", update.message);
                    setStatus("complete"); // Or handle error state
                    break;
            }
        };

        const handleClose = () => {
            console.log("[WebSocket] Connection closed");
            wsRef.current = null;
        };

        // Create WebSocket connection
        wsRef.current = createLiveSocket(
            sessionId,
            handleMessage,
            handleClose
        );

        // Cleanup on unmount or session change
        return () => {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
            if (thinkingIntervalRef.current) {
                clearInterval(thinkingIntervalRef.current);
                thinkingIntervalRef.current = null;
            }
        };
    }, [sessionId]);

    // Legacy simulation functions (kept for backward compatibility)
    const startSimulation = useCallback((newSessionId?: string) => {
        if (newSessionId) {
            setSessionId(newSessionId);
            setStatus("analyzing");
        }
        setCompletedAgents([]);
        setCurrentAgentIndex(-1);
    }, []);

    const resetSimulation = useCallback(() => {
        setSessionId(null);
        setStatus("idle");
        setCompletedAgents([]);
        setCurrentAgentIndex(-1);
        setCurrentThinkingPhrase("");
        setHitlCheckpoint(null);

        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        if (thinkingIntervalRef.current) {
            clearInterval(thinkingIntervalRef.current);
            thinkingIntervalRef.current = null;
        }
    }, []);

    // Dismiss HITL checkpoint
    const dismissCheckpoint = useCallback(() => {
        setHitlCheckpoint(null);
    }, []);

    return {
        status,
        currentAgentIndex,
        completedAgents,
        currentThinkingPhrase,
        startSimulation,
        resetSimulation,
        dismissCheckpoint,
        hitlCheckpoint,
        totalAgents: AGENTS_DATA.length
    };
};
