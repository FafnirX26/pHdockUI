"use client";

import { ChevronDown } from "lucide-react";

export default function Hero() {
  const scrollToInterface = () => {
    document.getElementById("molecule-interface")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative min-h-[60vh] flex items-center justify-center px-4 py-20 overflow-hidden">
      {/* Top blurred gradient background (papaya → onion purple) */}
      <div aria-hidden className="pointer-events-none absolute inset-x-0 -top-40 h-[60vh] -z-10">
        <div
          className="mx-auto h-full w-[120vw] max-w-none blur-3xl opacity-70"
          style={{
            background:
              "radial-gradient(120% 80% at 50% 0%, rgba(255, 205, 160, 0.65) 0%, rgba(255, 163, 102, 0.6) 22%, rgba(234, 88, 12, 0.55) 40%, rgba(168, 85, 247, 0.55) 70%, rgba(88, 28, 135, 0.6) 100%)",
            filter: "blur(60px)",
            transform: "translateY(-10%)",
          }}
        />
      </div>
      <div className="container mx-auto text-center max-w-4xl">
        <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          pH-aware Molecular Docking Suite
        </h1>
        <p className="text-lg md:text-xl text-gray-700 dark:text-gray-100 mb-8 leading-relaxed">
          Advanced computational toolkit integrating pH-dependent protonation states, 
          machine learning pKa prediction, and ensemble docking for accurate drug-target 
          interaction modeling at physiological conditions.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
          <button
            onClick={scrollToInterface}
            className="px-8 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors flex items-center gap-2"
          >
            Try Interactive Demo
            <ChevronDown size={20} />
          </button>
          <a
            href="https://github.com/DBD808/pHdockUI.git"
            target="_blank"
            rel="noopener noreferrer"
            className="px-8 py-3 border-2 border-gray-300 dark:border-gray-600 rounded-lg font-medium hover:border-gray-400 dark:hover:border-gray-500 transition-colors"
          >
            View on GitHub
          </a>
        </div>
        <div className="mt-12 grid grid-cols-3 gap-8 text-center">
          <div>
            <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">±0.5</div>
            <div className="text-sm text-gray-600 dark:text-gray-100">pKa RMSE</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">10x</div>
            <div className="text-sm text-gray-600 dark:text-gray-100">Faster than QM</div>
          </div>
          <div>
            <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">95%</div>
            <div className="text-sm text-gray-600 dark:text-gray-100">Accuracy</div>
          </div>
        </div>
      </div>
    </section>
  );
} 