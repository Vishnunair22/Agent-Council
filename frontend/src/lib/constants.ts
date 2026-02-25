import { AgentResult } from "@/types";

export type AgentDefinition = {
    id: string;
    name: string;
    role: string;
    desc: string;
    simulation: {
        result: string;
        confidence: number;
        thinking: string;
        thinkingPhrases: string[]; // Dynamic sub-tasks
    }
};

export const AGENTS_DATA: AgentDefinition[] = [
    {
        id: "1",
        name: "Image Integrity Expert",
        role: "Image Integrity",
        desc: "Detects manipulation, splicing, and GAN artifacts.",
        simulation: {
            result: "Noise distribution consistent with ISO 3200 sensor profile.",
            confidence: 99,
            thinking: "Analyzing sensor pattern noise (PRNU)...",
            thinkingPhrases: ["Checking error level analysis...", "Verifying quantization tables...", "Scanning for copy-move cloning..."]
        }
    },
    {
        id: "2",
        name: "Scene Reconstruction Expert",
        role: "Scene Reconstruction",
        desc: "Validates lighting, shadows, and perspective physics.",
        simulation: {
            result: "Shadow fall-off consistent with single key light at 45° elevation.",
            confidence: 96,
            thinking: "Calculating volumetric shadow vectors...",
            thinkingPhrases: ["Mapping vanishing points...", "Simulating light path bounces...", "Checking reflection consistency..."]
        }
    },
    {
        id: "3",
        name: "Object & Weapon Analyst",
        role: "Object & Weapon",
        desc: "Classifies weapons and generates confidence heatmaps.",
        simulation: {
            result: "Identified: Civilian Vehicle (Type A), Structure B (Residential).",
            confidence: 94,
            thinking: "Running YOLOv8 inference grid...",
            thinkingPhrases: ["Classifying obscure shapes...", "Comparing against ballistics DB...", "Generating heatmaps..."]
        }
    },
    {
        id: "4",
        name: "Temporal Video Analyst",
        role: "Temporal Video",
        desc: "Inspects frame consistency and deepfake markers.",
        simulation: {
            result: "Frame interval 33ms stable. Motion vectors align with camera track.",
            confidence: 98,
            thinking: "Mapping frame-to-frame pixel displacement...",
            thinkingPhrases: ["Analyzing optical flow...", "Detecting face-swapping artifacts...", "Checking audio-visual sync..."]
        }
    },
    {
        id: "5",
        name: "Metadata & Context Expert",
        role: "Metadata & Context",
        desc: "Analyzes EXIF data and device mismatch alerts.",
        simulation: {
            result: "GPS: 34.05°N, 118.24°W. Timestamp verified against solar positioning.",
            confidence: 99,
            thinking: "Cross-referencing satellite telemetry...",
            thinkingPhrases: ["Parsing EXIF/XMP tags...", "Verifying device signature...", "Checking weather at location..."]
        }
    }
];

// Computed helpers for backward compatibility if needed, or direct usage
export const MOCK_AGENTS = AGENTS_DATA.map(a => ({ name: a.name, role: a.role, desc: a.desc }));
