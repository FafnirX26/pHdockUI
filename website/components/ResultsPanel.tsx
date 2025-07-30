"use client";

import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { Loader2, Download, AlertCircle } from "lucide-react";
import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Type definitions
interface SitePka {
  pka: number;
  atom_index: number;
}

interface ProtonationState {
  state_id: number;
  charge: number;
  smiles: string;
  confidence: number;
}

interface DockingPose {
  state: number;
  score: number;
  confidence: number;
}

interface JobResults {
  molecule_info?: {
    smiles: string;
    molecular_weight: number;
  };
  pka_predictions?: {
    overall_pka: number;
    site_pkas: SitePka[];
    confidence: number;
  };
  protonation_states?: ProtonationState[];
  docking_results?: {
    poses: DockingPose[];
    best_score: number;
  };
}

interface JobData {
  status: string;
  progress: number;
  results?: JobResults;
  request?: {
    ph_value: number;
    smiles?: string;
  };
  error?: string;
}

interface ResultsPanelProps {
  jobId: string;
}

export default function ResultsPanel({ jobId }: ResultsPanelProps) {
  const [activeTab, setActiveTab] = useState<"pka" | "protonation" | "docking">("pka");

  const { data: jobResult, isLoading } = useQuery<JobData>({
    queryKey: ["job", jobId],
    queryFn: async () => {
      const response = await axios.get(`${API_URL}/api/jobs/${jobId}`);
      return response.data as JobData;
    },
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data?.status === "completed" || data?.status === "failed") {
        return false;
      }
      return 2000; // Poll every 2 seconds
    },
  });

  const downloadResults = () => {
    const results = JSON.stringify(jobResult?.results, null, 2);
    const blob = new Blob([results], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `phdock_results_${jobId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (isLoading || !jobResult) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="animate-spin mx-auto mb-4" size={48} />
          <p>Loading job status...</p>
        </div>
      </div>
    );
  }

  if (jobResult.status === "failed") {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center text-red-600">
          <AlertCircle className="mx-auto mb-4" size={48} />
          <p className="font-semibold">Analysis Failed</p>
          <p className="text-sm mt-2">{jobResult.error}</p>
        </div>
      </div>
    );
  }

  if (jobResult.status === "processing" || jobResult.status === "pending") {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="animate-spin mx-auto mb-4" size={48} />
          <p className="font-semibold">Processing molecule...</p>
          <div className="mt-4 w-64 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-500"
              style={{ width: `${(jobResult.progress || 0) * 100}%` }}
            />
          </div>
          <p className="text-sm mt-2">{Math.round((jobResult.progress || 0) * 100)}%</p>
        </div>
      </div>
    );
  }

  const results = jobResult.results;

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-semibold">Analysis Results</h3>
        <button
          onClick={downloadResults}
          className="flex items-center gap-2 px-4 py-2 text-sm bg-gray-100 dark:bg-gray-800 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
        >
          <Download size={16} />
          Download
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 dark:border-gray-700 mb-4">
        <button
          onClick={() => setActiveTab("pka")}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            activeTab === "pka"
              ? "border-blue-600 text-blue-600"
              : "border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
          }`}
        >
          pKa Predictions
        </button>
        <button
          onClick={() => setActiveTab("protonation")}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            activeTab === "protonation"
              ? "border-blue-600 text-blue-600"
              : "border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
          }`}
        >
          Protonation States
        </button>
        <button
          onClick={() => setActiveTab("docking")}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            activeTab === "docking"
              ? "border-blue-600 text-blue-600"
              : "border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
          }`}
        >
          Docking Results
        </button>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-auto">
        {activeTab === "pka" && (
          <div className="space-y-4">
            <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-medium mb-2">Global pKa</h4>
              <p className="text-2xl font-bold text-blue-600">
                {results?.pka_predictions?.overall_pka?.toFixed(2) || "N/A"}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Confidence: {((results?.pka_predictions?.confidence || 0) * 100).toFixed(1)}%
              </p>
            </div>
            
            {(results?.pka_predictions?.site_pkas?.length ?? 0) > 0 && (
              <div>
                <h4 className="font-medium mb-2">Site-specific pKa values</h4>
                <div className="space-y-2">
                  {results?.pka_predictions?.site_pkas?.map((site: SitePka, idx: number) => (
                    <div key={idx} className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-800 rounded">
                      <span className="text-sm">Site {idx + 1}</span>
                      <span className="font-mono">{site.pka?.toFixed(2) || "N/A"}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "protonation" && (
          <div className="space-y-4">
            {results?.protonation_states?.map((state: ProtonationState, idx: number) => (
              <div key={idx} className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium">State {state.state_id + 1}</h4>
                  <span className="text-sm bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-2 py-1 rounded">
                    Charge: {state.charge > 0 ? "+" : ""}{state.charge}
                  </span>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Confidence: {(state.confidence * 100).toFixed(1)}%
                </p>
                <p className="text-xs font-mono mt-2 break-all">{state.smiles}</p>
              </div>
            ))}
          </div>
        )}

        {activeTab === "docking" && (
          <div className="space-y-4">
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
              <h4 className="font-medium mb-2">Best Docking Score</h4>
              <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                {results?.docking_results?.best_score?.toFixed(2) || "N/A"} kcal/mol
              </p>
            </div>
            
            <div>
              <h4 className="font-medium mb-2">All Poses</h4>
              <div className="space-y-2">
                {results?.docking_results?.poses?.map((pose: DockingPose, idx: number) => (
                  <div key={idx} className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-800 rounded">
                    <span className="text-sm">Protonation State {pose.state + 1}</span>
                    <span className="font-mono">{pose.score.toFixed(2)} kcal/mol</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* CLI Command */}
      <div className="mt-4 p-3 bg-gray-900 dark:bg-gray-950 rounded text-xs">
        <p className="text-gray-400 mb-1">Reproduce via CLI:</p>
        <code className="text-green-400">
          python main.py --smiles &quot;{results?.molecule_info?.smiles}&quot; --ph {jobResult.request?.ph_value}
        </code>
      </div>
    </div>
  );
} 