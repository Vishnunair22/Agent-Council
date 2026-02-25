"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import { ChevronRight } from "lucide-react";
import Link from "next/link";
import { useRef, useEffect } from "react";
import { MOCK_AGENTS } from "@/lib/constants";
import { AgentIcon } from "@/components/ui/AgentIcon";

export default function LandingPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"],
  });

  // Force scroll to top on mount to ensure Hero is visible
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  // Tighter animation range [0, 0.2] for quick fade out.
  // Re-appears only at the very end [0.9, 1] to ensure "How it Works" is fully done.
  const heroScale = useTransform(scrollYProgress, [0, 0.2, 0.9, 1], [1, 0.8, 0.8, 1]);
  const heroOpacity = useTransform(scrollYProgress, [0, 0.2, 0.9, 1], [1, 0, 0, 1]);
  const heroBlur = useTransform(scrollYProgress, [0, 0.2, 0.9, 1], ["0px", "20px", "20px", "0px"]);

  // Content slides up naturally. 
  // Removed contentOpacity/Blur to ensure visibility.

  return (
    <div ref={containerRef} className="relative bg-black text-white">
      {/* --- Fixed Hero Section --- */}
      <motion.div
        style={{ scale: heroScale, opacity: heroOpacity, filter: heroBlur }}
        className="fixed top-0 left-0 w-full h-screen flex flex-col items-center justify-center z-0 pointer-events-none"
      >
        {/* Search Grid Background */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>
        <div className="absolute left-0 right-0 top-0 -z-10 m-auto h-[310px] w-[310px] rounded-full bg-cyan-500 opacity-20 blur-[100px]"></div>

        <header className="absolute top-0 w-full p-6 flex items-center justify-between border-b border-white/10 backdrop-blur-md z-50 pointer-events-auto">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-cyan-500 rounded flex items-center justify-center font-bold text-slate-900">FC</div>
            <span className="text-xl font-bold tracking-tight">Forensic Council</span>
          </div>
        </header>

        <div className="flex flex-col items-center justify-center text-center px-6 relative z-10 pointer-events-auto">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-2xl sm:text-4xl md:text-7xl font-bold max-w-4xl mb-4 md:mb-6 tracking-tighter text-white pb-2"
          >
            Multi-Agent Forensic Evidence Analysis System
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-white text-base sm:text-lg max-w-2xl mb-6 md:mb-10"
          >
            This system leverages multiple intelligent agents that analyze digital forensic evidence and compile those insights into a cohesive report.
          </motion.p>
          <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.4 }}>
            <Link href="/evidence" className="group relative px-8 py-4 bg-cyan-600 text-white font-bold rounded-full overflow-hidden transition-all hover:bg-cyan-500 inline-block">
              <span className="relative z-10 flex items-center">
                Begin Analysis <ChevronRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </span>
            </Link>
          </motion.div>
        </div>
      </motion.div>

      {/* --- Scrolling Content Section ---  */}
      {/* Added mb-[100vh] to allow scrolling past the footer to reveal hero again */}
      <div className="relative z-10 w-full mb-[100vh]" style={{ marginTop: "100vh" }}>
        <motion.div
          className="bg-black min-h-screen rounded-t-3xl border-t border-white/10 shadow-[0_-20px_40px_rgba(0,0,0,0.8)]"
          initial={{ y: 0 }}
        >
          {/* --- Meet the Agents ---  */}
          <section className="py-20 px-6 bg-slate-900/50 rounded-t-3xl">
            <h2 className="text-2xl md:text-3xl font-bold text-center mb-12">Meet the Agents</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6 max-w-7xl mx-auto">
              {MOCK_AGENTS.map((agent, i) => (
                <motion.div
                  key={i}
                  whileHover={{ y: -5 }}
                  className="p-6 rounded-2xl bg-slate-800/50 border border-white/5 flex flex-col items-center text-center hover:border-cyan-500/50 transition-colors"
                >
                  <div className="p-3 bg-cyan-500/10 text-cyan-500 rounded-xl mb-4">
                    <AgentIcon role={agent.role} />
                  </div>
                  <h3 className="font-semibold text-sm mb-2">{agent.name}</h3>
                  <p className="text-xs text-slate-500 leading-relaxed">{agent.desc}</p>
                </motion.div>
              ))}
            </div>
          </section>

          {/* --- How it Works ---  */}
          <section className="py-24 px-6 max-w-5xl mx-auto text-center bg-black">
            <h2 className="text-2xl md:text-3xl font-bold mb-16">How the Forensic Council Works</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
              {[
                { step: "01", title: "Evidence Intake", desc: "Upload digital media including CCTV, photos, or mobile evidence." },
                { step: "02", title: "Agent Consultation", desc: "Specialized agents analyze data and resolve contradictions." },
                { step: "03", title: "Final Verdict", desc: "A cohesive, legal-ready report is synthesized for review." }
              ].map((item, i) => (
                <div key={i} className="relative p-6">
                  <span className="text-5xl md:text-6xl font-black text-white/5 absolute -top-4 left-1/2 -translate-x-1/2 select-none">{item.step}</span>
                  <h4 className="text-xl font-bold mb-3 relative z-10">{item.title}</h4>
                  <p className="text-slate-400 text-sm relative z-10">{item.desc}</p>
                </div>
              ))}
            </div>
          </section>

          {/* --- Footer ---  */}
          <footer className="py-10 border-t border-white/5 text-center px-6 bg-black">
            <p className="text-slate-500 text-sm">
              Forensic Council is an academic project.
            </p>
          </footer>
        </motion.div>
      </div>
    </div>
  );
}
