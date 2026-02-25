"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { AGENTS_DATA } from "@/lib/constants";
import { AgentResult } from "@/types";

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

    const thinkingIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const mainTimerRef = useRef<NodeJS.Timeout | null>(null);

    const startSimulation = useCallback(() => {
        setStatus("analyzing");
        setCompletedAgents([]);
        setCurrentAgentIndex(-1);
    }, []);

    const resetSimulation = useCallback(() => {
        setStatus("idle");
        setCompletedAgents([]);
        setCurrentAgentIndex(-1);
        setCurrentThinkingPhrase("");

        if (thinkingIntervalRef.current) {
            clearInterval(thinkingIntervalRef.current);
            thinkingIntervalRef.current = null;
        }
        if (mainTimerRef.current) {
            clearTimeout(mainTimerRef.current);
            mainTimerRef.current = null;
        }
    }, []);

    // Main simulation orchestration
    useEffect(() => {
        console.log('[Simulation] Status:', status, 'Agent Index:', currentAgentIndex);

        if (status === "analyzing") {
            console.log('[Simulation] Starting analysis phase (4s)');
            // Step 1: Analyzing Evidence (4s)
            mainTimerRef.current = setTimeout(() => {
                console.log('[Simulation] Analysis complete, moving to initiating');
                setStatus("initiating");
            }, 4000);
        } else if (status === "initiating") {
            console.log('[Simulation] Initiating agents (1s)');
            // Step 2: Initiating Agents (brief transition)
            mainTimerRef.current = setTimeout(() => {
                console.log('[Simulation] Starting processing, agent 0');
                setStatus("processing");
                setCurrentAgentIndex(0);
            }, 1000);
        } else if (status === "processing" && currentAgentIndex >= 0 && currentAgentIndex < AGENTS_DATA.length) {
            // Step 3: Process each agent sequentially
            const agent = AGENTS_DATA[currentAgentIndex];
            console.log(`[Simulation] Processing agent ${currentAgentIndex}: ${agent.name}`);

            // Start thinking phrase rotation
            let phraseIndex = 0;
            setCurrentThinkingPhrase(agent.simulation.thinkingPhrases?.[0] || agent.simulation.thinking);

            if (agent.simulation.thinkingPhrases) {
                thinkingIntervalRef.current = setInterval(() => {
                    phraseIndex = (phraseIndex + 1) % agent.simulation.thinkingPhrases!.length;
                    setCurrentThinkingPhrase(agent.simulation.thinkingPhrases![phraseIndex]);
                }, 800);
            }

            // Agent analysis delay (4s)
            mainTimerRef.current = setTimeout(() => {
                console.log(`[Simulation] Agent ${currentAgentIndex} complete`);

                // Stop thinking rotation
                if (thinkingIntervalRef.current) {
                    clearInterval(thinkingIntervalRef.current);
                    thinkingIntervalRef.current = null;
                }

                // Create result
                const result: AgentResult = {
                    id: agent.id,
                    name: agent.name,
                    role: agent.role,
                    result: agent.simulation.result,
                    confidence: agent.simulation.confidence,
                    thinking: agent.simulation.thinking
                };

                // Add to completed agents
                setCompletedAgents(prev => {
                    console.log('[Simulation] Adding agent to completed list, prev length:', prev.length);
                    return [...prev, result];
                });
                onAgentComplete?.(result);
                playSound?.("agent");

                // Move to next agent or complete
                if (currentAgentIndex + 1 < AGENTS_DATA.length) {
                    console.log(`[Simulation] Moving to agent ${currentAgentIndex + 1}`);
                    setCurrentAgentIndex(prev => prev + 1);
                } else {
                    console.log('[Simulation] All agents complete');
                    setStatus("complete");
                    onComplete?.();
                    playSound?.("complete");
                }
            }, 4000);
        }

        // Cleanup
        return () => {
            if (mainTimerRef.current) {
                clearTimeout(mainTimerRef.current);
                mainTimerRef.current = null;
            }
            if (thinkingIntervalRef.current) {
                clearInterval(thinkingIntervalRef.current);
                thinkingIntervalRef.current = null;
            }
        };
    }, [status, currentAgentIndex]); // Removed callback dependencies to prevent timer cancellation

    return {
        status,
        currentAgentIndex,
        completedAgents,
        currentThinkingPhrase,
        startSimulation,
        resetSimulation,
        totalAgents: AGENTS_DATA.length
    };
};
