"use client";

import { useState, useEffect } from "react";
import { Report } from "@/types";
import { HistorySchema, ReportSchema } from "@/lib/schemas";

const HISTORY_KEY = "fc_history";
const CURRENT_REPORT_KEY = "fc_current_report";

export const useForensicData = () => {
    const [history, setHistory] = useState<Report[]>([]);
    const [currentReport, setCurrentReport] = useState<Report | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    // Load data on mount
    useEffect(() => {
        if (typeof window === 'undefined') return;

        try {
            const savedHistory = localStorage.getItem(HISTORY_KEY);
            const savedCurrent = localStorage.getItem(CURRENT_REPORT_KEY);

            if (savedHistory) {
                const parsed = JSON.parse(savedHistory);
                const validated = HistorySchema.safeParse(parsed);
                if (validated.success) {
                    setHistory(validated.data);
                }
            }

            if (savedCurrent) {
                const parsed = JSON.parse(savedCurrent);
                const validated = ReportSchema.safeParse(parsed);
                if (validated.success) {
                    setCurrentReport(validated.data);
                }
            }
        } catch (error) {
            console.error("Failed to load forensic data", error);
        } finally {
            setIsLoading(false);
        }
    }, []);

    const saveCurrentReport = (report: Report) => {
        if (typeof window === 'undefined') return;
        setCurrentReport(report);
        localStorage.setItem(CURRENT_REPORT_KEY, JSON.stringify(report));
    };

    const addToHistory = (report: Report) => {
        if (typeof window === 'undefined') return;
        const newHistory = [report, ...history];
        setHistory(newHistory);
        localStorage.setItem(HISTORY_KEY, JSON.stringify(newHistory));
    };

    const deleteFromHistory = (id: string) => {
        if (typeof window === 'undefined') return;
        const newHistory = history.filter(h => h.id !== id);
        setHistory(newHistory);
        localStorage.setItem(HISTORY_KEY, JSON.stringify(newHistory));
    };

    const clearHistory = () => {
        if (typeof window === 'undefined') return;
        setHistory([]);
        localStorage.removeItem(HISTORY_KEY);
    };

    // Client-side file validation
    const validateFile = (file: File): { valid: boolean; error?: string } => {
        const MAX_SIZE = 50 * 1024 * 1024; // 50MB
        const ALLOWED_TYPES = [
            "image/jpeg", "image/png", "image/webp", "image/gif",
            "video/mp4", "video/webm", "video/quicktime"
        ];

        if (file.size > MAX_SIZE) {
            return { valid: false, error: "File exceeds 50MB limit." };
        }
        if (!ALLOWED_TYPES.includes(file.type)) {
            return { valid: false, error: "Unsupported format. Use JPG, PNG, WEBP, MP4, or ARW/RAW." };
        }
        return { valid: true };
    };

    return {
        history,
        currentReport,
        isLoading,
        saveCurrentReport,
        addToHistory,
        deleteFromHistory,
        clearHistory,
        validateFile
    };
};
