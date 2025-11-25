"use client";

import { useState } from "react";
import { Upload, Loader2, ChevronRight, Settings } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import axios from "axios";
import MoleculeViewer from "./MoleculeViewer";
import ResultsPanel from "./ResultsPanel";
import ReceptorSearch from "./ReceptorSearch";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Type definitions
interface JobRequestData {
  smiles?: string;
  sdf_content?: string;
  ph_value: number;
  conformer_count: number;
  ensemble_size: number;
  quantum_fallback: boolean;
  docking_backend: string;
  receptor_id?: string;
}

interface JobResponse {
  job_id: string;
  status: string;
}

interface ExampleMolecule {
  name: string;
  smiles: string;
  description: string;
}

interface Receptor {
  id: string;
  name: string;
  pdb_id: string;
  description: string;
  resolution: string;
  use_case: string;
  available: boolean;
}

export default function MoleculeInterface() {
  const [smilesInput, setSmilesInput] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  
  // Advanced settings
  const [settings, setSettings] = useState({
    ph_value: 7.4,
    conformer_count: 10,
    ensemble_size: 5,
    quantum_fallback: false,
    docking_backend: "gnina",
    receptor_id: "mock"
  });

  // Fetch example molecules
  const { data: exampleMolecules } = useQuery<ExampleMolecule[]>({
    queryKey: ["example-molecules"],
    queryFn: async () => {
      const response = await axios.get(`${API_URL}/api/example-molecules`);
      return response.data;
    }
  });

  // Fetch available receptors
  useQuery<Receptor[]>({
    queryKey: ["receptors"],
    queryFn: async () => {
      const response = await axios.get(`${API_URL}/api/receptors`);
      return response.data;
    }
  });

  // Submit job mutation
  const submitJob = useMutation({
    mutationFn: async (data: JobRequestData) => {
      const response = await axios.post(`${API_URL}/api/jobs`, data);
      return response.data as JobResponse;
    },
    onSuccess: (data) => {
      setJobId(data.job_id);
    }
  });

  const handleSubmit = async () => {
    const jobData: JobRequestData = { ...settings };

    if (smilesInput) {
      jobData.smiles = smilesInput;
    } else if (selectedFile) {
      const fileContent = await selectedFile.text();
      jobData.sdf_content = fileContent;
    } else {
      alert("Please enter a SMILES string or upload a file");
      return;
    }

    submitJob.mutate(jobData);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setSmilesInput("");
    }
  };

  const handleExampleSelect = (smiles: string) => {
    setSmilesInput(smiles);
    setSelectedFile(null);
  };

  return (
    <section id="molecule-interface" className="py-20 px-4 bg-white dark:bg-gray-900">
      <div className="container mx-auto max-w-7xl">
        <h2 className="text-3xl font-bold text-center mb-12">Interactive pH Docking Analysis</h2>
        
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Panel */}
          <div className="bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-semibold mb-4">Input Molecule</h3>
            
            {/* SMILES Input */}
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">SMILES String</label>
              <input
                type="text"
                value={smilesInput}
                onChange={(e) => {
                  setSmilesInput(e.target.value);
                  setSelectedFile(null);
                }}
                placeholder="Enter SMILES notation..."
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-800"
              />
            </div>

            {/* File Upload */}
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">Or Upload File</label>
              <label className="flex items-center justify-center w-full h-32 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg cursor-pointer hover:border-gray-400 dark:hover:border-gray-500">
                <div className="text-center">
                  <Upload className="mx-auto mb-2 text-gray-400" size={24} />
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {selectedFile ? selectedFile.name : "Upload SDF/MOL2 file"}
                  </span>
                </div>
                <input
                  type="file"
                  onChange={handleFileUpload}
                  accept=".sdf,.mol2,.mol"
                  className="hidden"
                />
              </label>
            </div>

            {/* Example Molecules */}
            {exampleMolecules && (
              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Example Molecules</label>
                <div className="grid grid-cols-2 gap-2">
                  {exampleMolecules.map((molecule) => (
                    <button
                      key={molecule.name}
                      onClick={() => handleExampleSelect(molecule.smiles)}
                      className="px-3 py-2 text-sm bg-gray-100 dark:bg-gray-800 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                    >
                      {molecule.name}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Advanced Settings */}
            <div className="mb-4">
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-2 text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300"
              >
                <Settings size={16} />
                Advanced Settings
                <ChevronRight size={16} className={`transform transition-transform ${showAdvanced ? "rotate-90" : ""}`} />
              </button>
              
              {showAdvanced && (
                <div className="mt-4 space-y-3 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <div>
                    <label className="block text-sm font-medium mb-1">pH Value</label>
                    <input
                      type="number"
                      value={settings.ph_value}
                      onChange={(e) => setSettings({ ...settings, ph_value: parseFloat(e.target.value) })}
                      min="0"
                      max="14"
                      step="0.1"
                      className="w-full px-3 py-1 border border-gray-300 dark:border-gray-600 rounded dark:bg-gray-900"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Conformers</label>
                    <input
                      type="number"
                      value={settings.conformer_count}
                      onChange={(e) => setSettings({ ...settings, conformer_count: parseInt(e.target.value) })}
                      min="1"
                      max="100"
                      className="w-full px-3 py-1 border border-gray-300 dark:border-gray-600 rounded dark:bg-gray-900"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Ensemble Size</label>
                    <input
                      type="number"
                      value={settings.ensemble_size}
                      onChange={(e) => setSettings({ ...settings, ensemble_size: parseInt(e.target.value) })}
                      min="1"
                      max="20"
                      className="w-full px-3 py-1 border border-gray-300 dark:border-gray-600 rounded dark:bg-gray-900"
                    />
                  </div>
                  <div>
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={settings.quantum_fallback}
                        onChange={(e) => setSettings({ ...settings, quantum_fallback: e.target.checked })}
                        className="rounded"
                      />
                      <span className="text-sm font-medium">Enable Quantum Fallback</span>
                    </label>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Receptor</label>
                    <ReceptorSearch
                      value={settings.receptor_id}
                      onChange={(pdbId) => setSettings({ ...settings, receptor_id: pdbId })}
                      placeholder="Search RCSB PDB database..."
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Submit Button */}
            <button
              onClick={handleSubmit}
              disabled={submitJob.isPending || (!smilesInput && !selectedFile)}
              className="w-full py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
            >
              {submitJob.isPending ? (
                <>
                  <Loader2 className="animate-spin" size={20} />
                  Processing...
                </>
              ) : (
                <>
                  Run Analysis
                  <ChevronRight size={20} />
                </>
              )}
            </button>
          </div>

          {/* Results Panel */}
          <div className="bg-white dark:bg-gray-900 rounded-xl shadow-lg p-6">
            {jobId ? (
              <ResultsPanel jobId={jobId} />
            ) : (
              <div className="h-full flex items-center justify-center text-gray-500 dark:text-gray-400">
                <div className="text-center">
                  <MoleculeViewer className="mx-auto mb-4 opacity-20" />
                  <p>Submit a molecule to see results</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
} 