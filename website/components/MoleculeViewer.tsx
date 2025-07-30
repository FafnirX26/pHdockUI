import { Atom } from "lucide-react";

interface MoleculeViewerProps {
  className?: string;
  smiles?: string;
}

export default function MoleculeViewer({ className = "", smiles }: MoleculeViewerProps) {
  // In a real implementation, this would use RDKit.js to render the molecule
  // For now, we'll show a placeholder
  
  return (
    <div className={`flex items-center justify-center ${className}`} title={smiles ? `SMILES: ${smiles}` : "Molecule structure"}>
      <Atom size={64} className="text-blue-500" />
    </div>
  );
} 