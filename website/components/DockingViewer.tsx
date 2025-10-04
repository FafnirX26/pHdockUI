"use client";

import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { Loader2, AlertCircle } from "lucide-react";

interface DockingViewerProps {
  jobId: string;
  selectedStateId?: number;
  receptorId?: string;
}

// Extend Window interface for 3Dmol
declare global {
  interface Window {
    $3Dmol: any;
  }
}

export default function DockingViewer({ jobId, selectedStateId = 0, receptorId = "1ATP" }: DockingViewerProps) {
  const viewerRef = useRef<HTMLDivElement>(null);
  const [viewer, setViewer] = useState<any>(null);
  const [is3DmolLoaded, setIs3DmolLoaded] = useState(false);

  // Load 3Dmol.js library
  useEffect(() => {
    if (typeof window !== "undefined" && !window.$3Dmol) {
      const script = document.createElement("script");
      script.src = "https://3dmol.csb.pitt.edu/build/3Dmol-min.js";
      script.async = true;
      script.onload = () => setIs3DmolLoaded(true);
      document.body.appendChild(script);
    } else if (window.$3Dmol) {
      setIs3DmolLoaded(true);
    }
  }, []);

  // Fetch receptor PDB
  const { data: receptorData } = useQuery({
    queryKey: ["receptor-pdb", receptorId],
    queryFn: async () => {
      const response = await axios.get(`/api/receptors/${receptorId}/pdb`);
      return response.data;
    },
    enabled: is3DmolLoaded && !!receptorId,
  });

  // Fetch ligand PDB for selected protonation state
  const { data: ligandData, isLoading: isLigandLoading } = useQuery({
    queryKey: ["ligand-pdb", jobId, selectedStateId],
    queryFn: async () => {
      const response = await axios.get(`/api/jobs/${jobId}/ligand-pdb?state_id=${selectedStateId}`);
      return response.data;
    },
    enabled: is3DmolLoaded && !!jobId,
  });

  // Initialize viewer
  useEffect(() => {
    if (is3DmolLoaded && viewerRef.current && !viewer) {
      const config = { backgroundColor: "white" };
      const newViewer = window.$3Dmol.createViewer(viewerRef.current, config);
      setViewer(newViewer);
    }
  }, [is3DmolLoaded, viewer]);

  // Update visualization when data changes
  useEffect(() => {
    if (!viewer || !receptorData || !ligandData) return;

    viewer.clear();

    // Add receptor (protein)
    const receptorModel = viewer.addModel(receptorData.pdb_content, "pdb");
    viewer.setStyle(
      { model: receptorModel },
      {
        cartoon: { color: "spectrum" },
        line: { hidden: true }
      }
    );

    // Add ligand
    const ligandModel = viewer.addModel(ligandData.pdb_content, "pdb");
    viewer.setStyle(
      { model: ligandModel },
      {
        stick: { colorscheme: "greenCarbon", radius: 0.2 },
        sphere: { scale: 0.3 }
      }
    );

    // Center on ligand
    viewer.zoomTo({ model: ligandModel });
    viewer.zoom(0.8);
    viewer.render();

  }, [viewer, receptorData, ligandData]);

  if (!is3DmolLoaded || isLigandLoading) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg">
        <div className="text-center">
          <Loader2 className="animate-spin mx-auto mb-2" size={32} />
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {!is3DmolLoaded ? "Loading 3D viewer..." : "Loading structure..."}
          </p>
        </div>
      </div>
    );
  }

  if (!receptorData || !ligandData) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg">
        <div className="text-center text-gray-500">
          <AlertCircle className="mx-auto mb-2" size={32} />
          <p className="text-sm">Unable to load molecular structure</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative">
      <div
        ref={viewerRef}
        className="w-full h-full rounded-lg border border-gray-200 dark:border-gray-700"
        style={{ minHeight: "400px" }}
      />
      <div className="absolute top-2 left-2 bg-white/90 dark:bg-gray-900/90 px-3 py-1 rounded text-xs">
        <p className="font-medium">Receptor: {receptorId}</p>
        <p className="text-gray-600 dark:text-gray-400">Protonation State: {selectedStateId + 1}</p>
      </div>
      <div className="absolute bottom-2 right-2 bg-white/90 dark:bg-gray-900/90 px-3 py-1 rounded text-xs">
        <p className="text-gray-600 dark:text-gray-400">
          💡 Click and drag to rotate, scroll to zoom
        </p>
      </div>
    </div>
  );
}
