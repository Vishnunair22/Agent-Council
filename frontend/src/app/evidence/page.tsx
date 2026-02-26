"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileCheck, CheckCircle, RefreshCw, ArrowRight, Loader2, AlertCircle, X, ShieldAlert, ShieldCheck, ShieldX, ArrowRightCircle, Hand } from "lucide-react";
import { useRouter } from "next/navigation";
import { useForensicData } from "@/hooks/useForensicData";
import { useSimulation } from "@/hooks/useSimulation";
import { AgentResult } from "@/types";
import { AgentIcon } from "@/components/ui/AgentIcon";
import { AGENTS_DATA } from "@/lib/constants";
import { startInvestigation, submitHITLDecision, HITLCheckpoint, type HITLDecision } from "@/lib/api";

export default function EvidencePage() {
    const router = useRouter();
    const { saveCurrentReport, addToHistory, validateFile } = useForensicData();

    // --- State ---
    const [dragActive, setDragActive] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [caseId] = useState(() => `CASE-${Date.now()}`);
    const [sessionId, setSessionId] = useState<string | null>(null);

    const inputRef = useRef<HTMLInputElement>(null);
    const audioCtxRef = useRef<AudioContext | null>(null);

    // --- Audio ---
    const playSound = useCallback((type: "success" | "agent" | "complete" | "think") => {
        try {
            if (!audioCtxRef.current) {
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                if (AudioContext) {
                    audioCtxRef.current = new AudioContext();
                }
            }

            const ctx = audioCtxRef.current;
            if (!ctx) return;

            // Resume if suspended (browser auto-play policy)
            if (ctx.state === 'suspended') {
                ctx.resume().catch(console.error);
            }

            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);

            if (type === "success") {
                osc.type = "sine";
                osc.frequency.setValueAtTime(523.25, ctx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(1046.5, ctx.currentTime + 0.1);
                gain.gain.setValueAtTime(0.1, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.3);
            } else if (type === "agent") {
                osc.type = "triangle";
                osc.frequency.setValueAtTime(440, ctx.currentTime); // A4
                gain.gain.setValueAtTime(0.05, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.1);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.1);
            } else if (type === "think") {
                // Subtle low blip for thinking start
                osc.type = "sine";
                osc.frequency.setValueAtTime(220, ctx.currentTime);
                gain.gain.setValueAtTime(0.02, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.05);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.05);
            } else if (type === "complete") {
                osc.type = "sine";
                osc.frequency.setValueAtTime(440, ctx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.2);
                gain.gain.setValueAtTime(0.1, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.5);
            }
        } catch (e) {
            console.warn("Audio Context Error", e);
        }
    }, []);

    // Cleanup Audio Context on Unmount
    useEffect(() => {
        return () => {
            if (audioCtxRef.current) {
                audioCtxRef.current.close().catch(console.error);
                audioCtxRef.current = null;
            }
        };
    }, []);

    // --- Simulation Hook ---
    const {
        status,
        currentAgentIndex,
        completedAgents: simulationAgents,
        currentThinkingPhrase,
        startSimulation,
        resetSimulation,
        totalAgents,
        hitlCheckpoint,
        dismissCheckpoint
    } = useSimulation({
        playSound,
        onAgentComplete: (result) => {
            // Agent completed callback
        },
        onComplete: () => {
            // Simulation complete
        }
    });

    // --- Handlers ---
    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault(); e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
        else if (e.type === "dragleave") setDragActive(false);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault(); e.stopPropagation();
        setDragActive(false);
        setError(null);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const uploaded = e.dataTransfer.files[0];
            const check = validateFile(uploaded);
            if (check.valid) {
                setFile(uploaded);
                playSound("success");
            } else {
                setError(check.error || "Invalid file");
            }
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        e.preventDefault();
        setError(null);
        if (e.target.files && e.target.files[0]) {
            const uploaded = e.target.files[0];
            const check = validateFile(uploaded);
            if (check.valid) {
                setFile(uploaded);
                playSound("success");
            } else {
                setError(check.error || "Invalid file");
            }
        }
    };

    const startAnalysis = async (e: React.MouseEvent) => {
        e.stopPropagation();
        
        if (!file) {
            setError("No file selected");
            return;
        }

        try {
            setError(null);
            // Call the API to start investigation
            const result = await startInvestigation(
                file,
                caseId,
                "INVESTIGATOR-1"
            );
            
            // Get the session_id and pass it to the simulation hook
            const newSessionId = result.session_id;
            setSessionId(newSessionId);
            
            // Start simulation with the session ID (this connects to WebSocket)
            startSimulation(newSessionId);
            
            // Store session ID in sessionStorage for the result page
            sessionStorage.setItem('forensic_session_id', newSessionId);
            sessionStorage.setItem('forensic_case_id', caseId);
            sessionStorage.setItem('forensic_file_name', file.name);
        } catch (err) {
            console.error("Failed to start investigation:", err);
            setError(err instanceof Error ? err.message : "Failed to start analysis");
        }
    };

    const handleReset = (e: React.MouseEvent) => {
        e.stopPropagation();
        setFile(null);
        setError(null);
        setSessionId(null);
        sessionStorage.removeItem('forensic_session_id');
        sessionStorage.removeItem('forensic_case_id');
        sessionStorage.removeItem('forensic_file_name');
        resetSimulation();
    };

    const acceptAnalysis = (e: React.MouseEvent) => {
        e.stopPropagation();

        // Strip extra properties (thinking, icon) for storage
        const cleanAgents: AgentResult[] = simulationAgents.map((a: AgentResult) => ({
            id: a.id,
            name: a.name,
            role: a.role,
            result: a.result,
            confidence: a.confidence
        }));

        const reportData = {
            id: sessionId || Date.now().toString(),
            fileName: file?.name || "Unknown File",
            timestamp: new Date().toISOString(),
            agents: cleanAgents,
            summary: `Analyzed ${cleanAgents.length} forensic indicators.`,
            sessionId: sessionId,
            caseId: caseId
        };

        saveCurrentReport(reportData);
        addToHistory(reportData);
        router.push("/result");
    };

    // --- HITL Decision Handler ---
    const handleHITLDecision = async (decision: HITLDecision) => {
        if (!hitlCheckpoint || !sessionId) {
            console.error("No HITL checkpoint or session ID available");
            return;
        }

        try {
            await submitHITLDecision({
                session_id: sessionId,
                checkpoint_id: hitlCheckpoint.checkpoint_id,
                agent_id: hitlCheckpoint.agent_id,
                decision: decision,
                note: `Decision: ${decision}`
            });
            
            // Dismiss the checkpoint to continue
            dismissCheckpoint();
        } catch (err) {
            console.error("Failed to submit HITL decision:", err);
            setError(err instanceof Error ? err.message : "Failed to submit decision");
        }
    };


    return (
        <div
            className="flex min-h-screen flex-col items-center justify-center bg-black text-white p-6 cursor-pointer overflow-hidden"
            onClick={() => router.push('/')}
        >
            {/* Background Gradients */}
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900/50 via-black to-black -z-50" />

            <header className="absolute top-0 w-full p-6 flex items-center justify-between border-b border-white/10 backdrop-blur-md z-50">
                <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-emerald-500 rounded flex items-center justify-center font-bold text-slate-900">FC</div>
                    <span className="text-xl font-bold tracking-tight">Forensic Council</span>
                </div>
            </header>

            <main
                className="w-full max-w-4xl flex flex-col items-center gap-8 cursor-auto z-10"
                onClick={(e) => e.stopPropagation()}
            >
                {/* --- Header --- */}
                <motion.div
                    layout
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-center w-full max-w-md"
                >
                    <h1 className="text-2xl md:text-4xl font-bold mb-2">Evidence Intake</h1>
                    <p className="text-slate-400 h-6 min-h-[1.5rem] flex items-center justify-center gap-2 mb-4">
                        {status === "idle" && "Upload digital media for forensic analysis."}
                        {status === "analyzing" && (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" /> Analyzing Evidence...
                            </>
                        )}
                        {status === "initiating" && (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin text-emerald-500" /> Initiating Agents...
                            </>
                        )}
                        {status === "processing" && currentAgentIndex >= 0 && (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin text-emerald-500" />
                                <span className="animate-pulse text-emerald-500 font-mono">
                                    {AGENTS_DATA[currentAgentIndex]?.name} &gt; {currentThinkingPhrase || "PROCESSING"}
                                </span>
                            </>
                        )}
                        {status === "complete" && "Analysis Complete. Review findings below."}
                    </p>

                    {/* Progress Bar */}
                    {(status === "processing" || status === "analyzing" || status === "initiating") && (
                        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mb-8">
                            <motion.div
                                className="bg-emerald-500 h-full"
                                initial={{ width: 0 }}
                                animate={{
                                    width: status === "analyzing" ? "10%" : status === "initiating" ? "15%" : `${((currentAgentIndex + 1) / totalAgents) * 100}%`
                                }}
                                transition={{ duration: 0.5, ease: "easeInOut" }}
                            />
                        </div>
                    )}
                </motion.div>


                {/* --- Upload Zone (Hidden when analyzing starts) --- */}
                <AnimatePresence>
                    {status === "idle" && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: "auto" }}
                            exit={{ opacity: 0, height: 0, overflow: "hidden" }}
                            className="w-full max-w-2xl"
                        >
                            <div
                                className={`relative w-full h-48 md:h-64 rounded-3xl border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center cursor-pointer overflow-hidden ${dragActive
                                    ? "border-emerald-500 bg-emerald-500/10"
                                    : "border-white/20 hover:border-white/40 hover:bg-white/5"
                                    }`}
                                onDragEnter={handleDrag}
                                onDragLeave={handleDrag}
                                onDragOver={handleDrag}
                                onDrop={handleDrop}
                                onClick={() => inputRef.current?.click()}
                            >
                                <input
                                    ref={inputRef}
                                    type="file"
                                    className="hidden"
                                    onChange={handleChange}
                                    accept="image/*,video/*"
                                />

                                <AnimatePresence mode="wait">
                                    {file ? (
                                        <motion.div
                                            key="file-uploaded"
                                            initial={{ opacity: 0, scale: 0.8 }}
                                            animate={{ opacity: 1, scale: 1 }}
                                            exit={{ opacity: 0, scale: 0.8 }}
                                            className="flex flex-col items-center text-emerald-400"
                                        >
                                            <FileCheck className="w-16 h-16 mb-4" />
                                            <p className="font-medium text-lg text-white">{file.name}</p>
                                            <p className="text-sm text-emerald-400/70">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                            <p className="mt-4 text-xs text-slate-500">Click to replace</p>
                                        </motion.div>
                                    ) : (
                                        <motion.div
                                            key="upload-prompt"
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            exit={{ opacity: 0 }}
                                            className="flex flex-col items-center text-slate-400"
                                        >
                                            <Upload className={`w-12 h-12 mb-4 transition-transform duration-300 ${dragActive ? 'scale-110 text-emerald-500' : ''}`} />
                                            <p className="font-medium">Drag & drop evidence here</p>
                                            <p className="text-sm mt-2 opacity-60">or click to browse</p>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>

                            {file && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="flex justify-center mt-6 w-full md:w-auto"
                                >
                                    <button
                                        onClick={startAnalysis}
                                        className="w-full md:w-auto px-8 py-3 bg-emerald-600 hover:bg-emerald-500 text-white rounded-full font-bold transition-all shadow-lg hover:shadow-emerald-500/20"
                                        aria-label="Start Analysis"
                                    >
                                        Initiate Analysis
                                    </button>
                                </motion.div>
                            )}

                            {/* Error Message */}
                            <AnimatePresence>
                                {error && (
                                    <motion.div
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        exit={{ opacity: 0, y: -10 }}
                                        className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm flex items-center gap-2"
                                    >
                                        <AlertCircle className="w-4 h-4" /> {error}
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </motion.div>
                    )}
                </AnimatePresence>


                {/* --- Analysis Grid --- */}
                {(status === "processing" || status === "complete") && (
                    <motion.div
                        className="w-full grid grid-cols-1 md:grid-cols-2 gap-4"
                        layout
                    >
                        <AnimatePresence>
                            {simulationAgents.map((agent, i) => (
                                <motion.div
                                    key={agent.id}
                                    initial={{ opacity: 0, x: -20, filter: "blur(10px)" }}
                                    animate={{ opacity: 1, x: 0, filter: "blur(0px)" }}
                                    // Stagger effect naturally happens because items are added over time
                                    className="p-5 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-lg flex items-start space-x-4 shadow-xl will-change-transform"
                                >
                                    <div className="p-3 bg-emerald-500/20 text-emerald-400 rounded-xl">
                                        <AgentIcon role={agent.role} />
                                    </div>
                                    <div>
                                        <h3 className="font-bold text-white relative flex items-center gap-2">
                                            {agent.name}
                                            <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-white/10 uppercase tracking-wider">{agent.role}</span>
                                        </h3>
                                        <p className="text-sm text-slate-300 mt-1 leading-relaxed">{agent.result}</p>
                                        <div className="mt-3 flex items-center text-emerald-500 text-xs font-mono font-bold">
                                            <CheckCircle className="w-3 h-3 mr-1" /> CONFIDENCE: {agent.confidence}%
                                        </div>
                                    </div>
                                </motion.div>
                            ))}

                            {/* Thinking Card (Current Agent) - Only show if current agent is processing */}
                            {status === "processing" && currentAgentIndex >= 0 && currentAgentIndex < totalAgents && (
                                <motion.div
                                    key="thinking"
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    exit={{ opacity: 0, scale: 0.95 }}
                                    className="p-5 rounded-2xl bg-white/5 border border-white/5 border-dashed flex items-center justify-center space-x-3 text-slate-500 min-h-[140px] will-change-transform"
                                >
                                    <div className="w-5 h-5 border-2 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin" />
                                    <span className="animate-pulse font-mono uppercase tracking-widest text-xs">
                                        {AGENTS_DATA[currentAgentIndex]?.name} Analysing...
                                    </span>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </motion.div>
                )}

                {/* --- Completion Actions --- */}
                {status === "complete" && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        className="flex flex-col md:flex-row gap-4 mt-4 w-full md:w-auto"
                    >
                        <button
                            onClick={handleReset}
                            className="w-full md:w-auto px-6 py-3 rounded-full border border-white/20 hover:bg-white/10 text-white font-semibold transition-all flex items-center justify-center gap-2"
                        >
                            <RefreshCw className="w-4 h-4" /> Analyse Again
                        </button>
                        <button
                            onClick={acceptAnalysis}
                            className="w-full md:w-auto px-8 py-3 bg-white text-black hover:bg-slate-200 rounded-full font-bold transition-all shadow-[0_0_20px_rgba(255,255,255,0.3)] flex items-center justify-center gap-2"
                        >
                            Accept Analysis <ArrowRight className="w-4 h-4" />
                        </button>
                    </motion.div>
                )}

            </main>

            {/* --- HITL Modal */}
            <AnimatePresence>
                {hitlCheckpoint && sessionId && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
                        onClick={() => {}}
                    >
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            className="bg-slate-900 border border-amber-500/50 rounded-3xl p-6 md:p-8 max-w-2xl w-full shadow-2xl shadow-amber-500/20"
                            onClick={(e) => e.stopPropagation()}
                        >
                            {/* Header */}
                            <div className="flex items-center justify-between mb-6">
                                <div className="flex items-center gap-3">
                                    <div className="p-3 bg-amber-500/20 rounded-xl">
                                        <ShieldAlert className="w-8 h-8 text-amber-500" />
                                    </div>
                                    <div>
                                        <h2 className="text-xl font-bold text-white">Human-in-the-Loop Checkpoint</h2>
                                        <p className="text-sm text-amber-400/70">{hitlCheckpoint!.agent_name} requires approval</p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => dismissCheckpoint()}
                                    className="p-2 hover:bg-white/10 rounded-full transition-colors"
                                >
                                    <X className="w-5 h-5 text-slate-400" />
                                </button>
                            </div>

                            {/* Brief Content */}
                            <div className="bg-black/40 rounded-2xl p-5 mb-6 border border-white/5">
                                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Investigator Brief</h3>
                                <p className="text-slate-300 leading-relaxed whitespace-pre-wrap">{hitlCheckpoint!.brief_text}</p>
                            </div>

                            {/* Decision Buttons */}
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                <button
                                    onClick={() => handleHITLDecision("APPROVE")}
                                    className="flex flex-col items-center gap-2 p-4 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 rounded-xl transition-all group"
                                >
                                    <ShieldCheck className="w-6 h-6 text-emerald-500 group-hover:scale-110 transition-transform" />
                                    <span className="text-sm font-semibold text-emerald-400">Approve</span>
                                </button>

                                <button
                                    onClick={() => handleHITLDecision("REDIRECT")}
                                    className="flex flex-col items-center gap-2 p-4 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 rounded-xl transition-all group"
                                >
                                    <ArrowRightCircle className="w-6 h-6 text-blue-500 group-hover:scale-110 transition-transform" />
                                    <span className="text-sm font-semibold text-blue-400">Redirect</span>
                                </button>

                                <button
                                    onClick={() => handleHITLDecision("OVERRIDE")}
                                    className="flex flex-col items-center gap-2 p-4 bg-orange-500/10 hover:bg-orange-500/20 border border-orange-500/30 rounded-xl transition-all group"
                                >
                                    <Hand className="w-6 h-6 text-orange-500 group-hover:scale-110 transition-transform" />
                                    <span className="text-sm font-semibold text-orange-400">Override</span>
                                </button>

                                <button
                                    onClick={() => handleHITLDecision("TERMINATE")}
                                    className="flex flex-col items-center gap-2 p-4 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 rounded-xl transition-all group"
                                >
                                    <ShieldX className="w-6 h-6 text-red-500 group-hover:scale-110 transition-transform" />
                                    <span className="text-sm font-semibold text-red-400">Terminate</span>
                                </button>
                            </div>

                            <p className="text-center text-xs text-slate-500 mt-4">
                                Choose an action to continue the analysis pipeline
                            </p>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
