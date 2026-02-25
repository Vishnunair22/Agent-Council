"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Clock, Trash2, AlertTriangle, ChevronDown, Download, FileText, FileJson, FileType } from "lucide-react";
import { useRouter } from "next/navigation";
import clsx from "clsx";
import { AgentIcon } from "@/components/ui/AgentIcon";
import { useForensicData } from "@/hooks/useForensicData";

export default function ResultPage() {
    const router = useRouter();
    const [activeTab, setActiveTab] = useState<"result" | "history">("result");

    // Use Hook
    const { history, currentReport, deleteFromHistory, clearHistory, isLoading } = useForensicData();

    // Refinement States
    const [showAgents, setShowAgents] = useState(true);
    const [exportOpen, setExportOpen] = useState(false);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    // Prevent hydration mismatch
    if (!mounted) return null;

    // --- Handlers ---
    const handleDeleteOne = (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        deleteFromHistory(id);
    };

    const handleClearAll = () => {
        clearHistory();
    };

    const formatDate = (isoString: string) => {
        return new Date(isoString).toLocaleString();
    };

    return (
        <div className="min-h-screen bg-black text-white p-6 pb-20">
            {/* --- Background --- */}
            <div className="fixed inset-0 bg-[radial-gradient(circle_at_top,_var(--tw-gradient-stops))] from-slate-900/40 via-black to-black -z-50" />

            {/* Header Removed as per request for cleaner UI */}
            <div className="pt-10" />

            <main className="max-w-5xl mx-auto">
                {/* --- Tabs --- */}
                <div className="flex space-x-6 mb-8 border-b border-white/10">
                    <button
                        onClick={() => setActiveTab("result")}
                        className={clsx(
                            "pb-3 text-lg font-medium transition-all relative",
                            activeTab === "result" ? "text-emerald-500" : "text-slate-500 hover:text-slate-300"
                        )}
                    >
                        Current Analysis
                        {activeTab === "result" && (
                            <motion.div layoutId="tab" className="absolute bottom-0 left-0 w-full h-0.5 bg-emerald-500" />
                        )}
                    </button>
                    <button
                        onClick={() => setActiveTab("history")}
                        className={clsx(
                            "pb-3 text-lg font-medium transition-all relative",
                            activeTab === "history" ? "text-emerald-500" : "text-slate-500 hover:text-slate-300"
                        )}
                    >
                        History
                        {activeTab === "history" && (
                            <motion.div layoutId="tab" className="absolute bottom-0 left-0 w-full h-0.5 bg-emerald-500" />
                        )}
                    </button>
                </div>

                {/* --- Content --- */}
                <AnimatePresence mode="wait">
                    {activeTab === "result" ? (
                        <motion.div
                            key="result"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="space-y-8"
                        >
                            {isLoading ? (
                                <div className="p-12 text-center text-slate-500">
                                    <div className="w-8 h-8 border-2 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin mx-auto mb-4" />
                                    <p>Loading forensic data...</p>
                                </div>
                            ) : currentReport ? (
                                <>
                                    {/* Summary Card */}
                                    <div className="p-5 md:p-8 rounded-3xl bg-gradient-to-br from-emerald-900/20 to-slate-900/40 border border-emerald-500/20 shadow-2xl">
                                        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
                                            <div>
                                                <h2 className="text-2xl md:text-3xl font-bold text-white mb-2 break-all">{currentReport.fileName}</h2>
                                                <p className="text-slate-400 font-mono text-sm">{formatDate(currentReport.timestamp)}</p>
                                            </div>
                                            <div className="flex flex-wrap gap-3">
                                                {/* Export Dropdown */}
                                                <div className="relative">
                                                    <button
                                                        onClick={() => setExportOpen(!exportOpen)}
                                                        className="flex items-center space-x-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-full text-slate-300 transition-colors border border-white/10"
                                                    >
                                                        <Download className="w-4 h-4" />
                                                        <span>Export</span>
                                                        <ChevronDown className={`w-4 h-4 transition-transform ${exportOpen ? 'rotate-180' : ''}`} />
                                                    </button>

                                                    <AnimatePresence>
                                                        {exportOpen && (
                                                            <motion.div
                                                                initial={{ opacity: 0, y: 10 }}
                                                                animate={{ opacity: 1, y: 0 }}
                                                                exit={{ opacity: 0, y: 10 }}
                                                                className="absolute right-0 mt-2 w-48 bg-slate-800 rounded-xl shadow-xl border border-white/10 overflow-hidden z-20"
                                                            >
                                                                <button className="w-full px-4 py-3 text-left bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center gap-3 transition-colors text-sm">
                                                                    <FileText className="w-4 h-4" /> PDF Report
                                                                </button>
                                                                <button className="w-full px-4 py-3 text-left bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center gap-3 transition-colors text-sm border-t border-white/5">
                                                                    <FileType className="w-4 h-4" /> DOCX Format
                                                                </button>
                                                                <button
                                                                    onClick={() => {
                                                                        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentReport, null, 2));
                                                                        const downloadAnchorNode = document.createElement('a');
                                                                        downloadAnchorNode.setAttribute("href", dataStr);
                                                                        downloadAnchorNode.setAttribute("download", `forensic_report_${currentReport.id}.json`);
                                                                        document.body.appendChild(downloadAnchorNode);
                                                                        downloadAnchorNode.click();
                                                                        downloadAnchorNode.remove();
                                                                        setExportOpen(false);
                                                                    }}
                                                                    className="w-full px-4 py-3 text-left bg-slate-800 hover:bg-slate-700 text-emerald-400 flex items-center gap-3 transition-colors text-sm border-t border-white/5"
                                                                >
                                                                    <FileJson className="w-4 h-4" /> JSON Data
                                                                </button>
                                                            </motion.div>
                                                        )}
                                                    </AnimatePresence>
                                                </div>

                                                {/* Badge Removed as per request */}
                                            </div>
                                        </div>
                                        <div className="space-y-4">
                                            <h3 className="text-xl font-semibold text-emerald-400">Executive Summary</h3>
                                            <p className="text-base md:text-lg text-slate-200 leading-relaxed font-light">{currentReport.summary}</p>
                                            <p className="text-slate-400 leading-relaxed">
                                                The Council has aggregated data from all available forensic layers. The following key signals were extracted and cross-referenced:
                                            </p>

                                            <div className="bg-slate-900/50 rounded-xl p-4 border border-white/5 mt-4">
                                                <h4 className="text-xs md:text-sm font-bold text-slate-300 mb-3 uppercase tracking-wider">Key Signals Detected</h4>
                                                <ul className="space-y-2">
                                                    {currentReport.agents.map((agent: any, i: number) => (
                                                        <li key={i} className="flex items-start text-sm text-slate-400 gap-2">
                                                            <div className="mt-1.5 w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0" />
                                                            <span>
                                                                <strong className="text-slate-300">{agent.name}:</strong> {agent.result}
                                                            </span>
                                                        </li>
                                                    ))}
                                                </ul>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Toggle for Individual Agents */}
                                    <div className="flex justify-center">
                                        <button
                                            onClick={() => setShowAgents(!showAgents)}
                                            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
                                        >
                                            <span>{showAgents ? "Hide" : "See"} Individual Agent Analysis</span>
                                            <ChevronDown className={`w-4 h-4 transition-transform ${showAgents ? 'rotate-180' : ''}`} />
                                        </button>
                                    </div>

                                    {/* Agent Breakdown (Collapsible) */}
                                    <AnimatePresence>
                                        {showAgents && (
                                            <motion.div
                                                initial={{ opacity: 0, height: 0 }}
                                                animate={{ opacity: 1, height: "auto" }}
                                                exit={{ opacity: 0, height: 0 }}
                                                className="overflow-hidden"
                                            >
                                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-4">
                                                    {currentReport.agents.map((agent: any, i: number) => (
                                                        <div key={i} className="p-5 rounded-2xl bg-white/5 border border-white/5 hover:border-white/10 transition-colors">
                                                            <div className="flex items-center gap-3 mb-3">
                                                                <div className="p-2 bg-slate-800 rounded-lg text-emerald-400">
                                                                    <AgentIcon role={agent.role} />
                                                                </div>
                                                                <div>
                                                                    <h4 className="font-bold text-sm">{agent.name}</h4>
                                                                    <p className="text-xs text-slate-500">{agent.role}</p>
                                                                </div>
                                                            </div>
                                                            <p className="text-sm text-slate-400 leading-relaxed mb-3 h-14 overflow-hidden">{agent.result}</p>
                                                            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                                                                <div className="bg-emerald-500 h-full" style={{ width: `${agent.confidence}%` }} />
                                                            </div>
                                                            <p className="text-right text-[10px] text-emerald-500 font-mono mt-1">{agent.confidence}% CONFIDENCE</p>
                                                        </div>
                                                    ))}
                                                </div>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>

                                    {/* Return to Home Action - Bottom */}
                                    <div className="flex justify-center pt-8 border-t border-white/10 mt-8">
                                        <button
                                            onClick={() => router.push('/')}
                                            className="w-full md:w-auto px-8 py-3 bg-white text-black hover:bg-slate-200 rounded-full font-bold transition-all shadow-lg flex items-center justify-center gap-2"
                                        >
                                            Return to Home
                                        </button>
                                    </div>

                                </>
                            ) : (
                                <div className="text-center py-20 text-slate-500">
                                    <AlertTriangle className="w-12 h-12 mx-auto mb-4 opacity-50" />
                                    <p>No recent analysis found.</p>
                                    <button onClick={() => router.push('/evidence')} className="mt-4 text-emerald-500 hover:underline">Start New Analysis</button>
                                </div>
                            )}
                        </motion.div>
                    ) : (
                        /* --- History Tab Content (Same as before) --- */
                        <motion.div
                            key="history"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                        >
                            <div className="flex justify-between items-center mb-6">
                                <h3 className="text-slate-400 font-medium">Recorded Sessions ({history.length})</h3>
                                {history.length > 0 && (
                                    <button onClick={handleClearAll} className="text-red-400 hover:text-red-300 text-sm flex items-center transition-colors">
                                        <Trash2 className="w-4 h-4 mr-2" /> Clear All Activity
                                    </button>
                                )}
                            </div>

                            <div className="space-y-4">
                                <AnimatePresence>
                                    {history.length > 0 ? history.map((item: any) => (
                                        <motion.div
                                            key={item.id}
                                            layout
                                            initial={{ opacity: 0, scale: 0.95 }}
                                            animate={{ opacity: 1, scale: 1 }}
                                            exit={{ opacity: 0, height: 0, marginBottom: 0, overflow: "hidden" }}
                                            className="p-5 rounded-2xl bg-white/5 border border-white/5 flex items-center justify-between group"
                                        >
                                            <div className="flex items-center gap-4">
                                                <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center text-slate-400 font-mono font-bold">
                                                    FC
                                                </div>
                                                <div>
                                                    <h4 className="font-bold text-white">{item.fileName}</h4>
                                                    <div className="flex items-center text-xs text-slate-500 gap-2">
                                                        <Clock className="w-3 h-3" /> {formatDate(item.timestamp)}
                                                    </div>
                                                </div>
                                            </div>

                                            <div className="flex items-center gap-6">
                                                <div className="text-right hidden sm:block">
                                                    <p className="text-emerald-400 text-sm font-bold">Safe</p>
                                                    <p className="text-xs text-slate-600">Avg. 94%</p>
                                                </div>
                                                <button
                                                    onClick={(e) => handleDeleteOne(item.id, e)}
                                                    className="p-2 text-slate-600 hover:text-red-400 hover:bg-red-500/10 rounded-full transition-all"
                                                    title="Delete Entry"
                                                >
                                                    <Trash2 className="w-4 h-4" />
                                                </button>
                                            </div>
                                        </motion.div>
                                    )) : (
                                        <div className="text-center py-10 text-slate-600">History is empty.</div>
                                    )}
                                </AnimatePresence>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

            </main>
        </div>
    );
}
