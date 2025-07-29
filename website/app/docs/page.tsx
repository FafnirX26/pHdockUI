import { Terminal, Download, Book, Github, ExternalLink } from "lucide-react";
import Link from "next/link";

export default function DocsPage() {
  return (
    <div className="min-h-screen py-20 px-4">
      <div className="container mx-auto max-w-4xl">
        <h1 className="text-4xl font-bold mb-8">Documentation</h1>

        {/* Quick Start */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
            <Terminal className="text-blue-600" size={24} />
            Quick Start
          </h2>
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6 space-y-4">
            <div>
              <h3 className="font-medium mb-2">Installation</h3>
              <pre className="bg-gray-900 dark:bg-gray-950 text-gray-100 p-4 rounded overflow-x-auto">
                <code>{`pip install phdockui

# Or install from source
git clone https://github.com/yourusername/pHdockUI.git
cd pHdockUI
pip install -e .`}</code>
              </pre>
            </div>

            <div>
              <h3 className="font-medium mb-2">Basic Usage</h3>
              <pre className="bg-gray-900 dark:bg-gray-950 text-gray-100 p-4 rounded overflow-x-auto">
                <code>{`from phdockui import pHDocking

# Initialize the docking system
docker = pHDocking()

# Run pH-aware docking
results = docker.dock(
    ligand_smiles="CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    receptor_pdb="receptor.pdb",
    ph=7.4
)

# Access results
print(f"Best docking score: {results.best_score}")
print(f"Predicted pKa: {results.pka_values}")`}</code>
              </pre>
            </div>

            <div>
              <h3 className="font-medium mb-2">Command Line Interface</h3>
              <pre className="bg-gray-900 dark:bg-gray-950 text-gray-100 p-4 rounded overflow-x-auto">
                <code>{`# Basic docking
phdock --smiles "CC(=O)Oc1ccccc1C(=O)O" --receptor receptor.pdb --ph 7.4

# With advanced options
phdock --sdf ligand.sdf \\
       --receptor receptor.pdb \\
       --ph 6.5 \\
       --conformers 50 \\
       --ensemble-size 10 \\
       --output results.json`}</code>
              </pre>
            </div>
          </div>
        </section>

        {/* API Reference */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
            <Book className="text-purple-600" size={24} />
            API Reference
          </h2>
          <div className="space-y-6">
            <div className="border-l-4 border-blue-500 pl-4">
              <h3 className="font-mono font-semibold">pHDocking()</h3>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                Main class for pH-aware molecular docking
              </p>
              <div className="mt-4 space-y-2">
                <h4 className="font-medium">Methods:</h4>
                <ul className="space-y-1 text-sm">
                  <li><code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">dock(ligand, receptor, ph)</code> - Run docking analysis</li>
                  <li><code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">predict_pka(molecule)</code> - Predict pKa values</li>
                  <li><code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">generate_protonation_states(molecule, ph)</code> - Generate states</li>
                </ul>
              </div>
            </div>

            <div className="border-l-4 border-purple-500 pl-4">
              <h3 className="font-mono font-semibold">ConformerGenerator</h3>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                Generate 3D conformers for molecules
              </p>
            </div>

            <div className="border-l-4 border-green-500 pl-4">
              <h3 className="font-mono font-semibold">EnsembleModel</h3>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                Machine learning ensemble for pKa prediction
              </p>
            </div>
          </div>
        </section>

        {/* Resources */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
            <Download className="text-green-600" size={24} />
            Resources
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            <a
              href="https://github.com/yourusername/pHdockUI"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              <Github size={24} />
              <div>
                <h3 className="font-medium">GitHub Repository</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">Source code and issues</p>
              </div>
              <ExternalLink size={16} className="ml-auto" />
            </a>

            <a
              href="#"
              className="flex items-center gap-3 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              <Book size={24} />
              <div>
                <h3 className="font-medium">Example Notebooks</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">Jupyter tutorials</p>
              </div>
              <ExternalLink size={16} className="ml-auto" />
            </a>

            <a
              href="#"
              className="flex items-center gap-3 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              <Download size={24} />
              <div>
                <h3 className="font-medium">Test Dataset</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">Sample molecules</p>
              </div>
              <ExternalLink size={16} className="ml-auto" />
            </a>

            <a
              href="#"
              className="flex items-center gap-3 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              <Terminal size={24} />
              <div>
                <h3 className="font-medium">Docker Image</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">Pre-configured container</p>
              </div>
              <ExternalLink size={16} className="ml-auto" />
            </a>
          </div>
        </section>

        {/* FAQ */}
        <section>
          <h2 className="text-2xl font-semibold mb-4">Frequently Asked Questions</h2>
          <div className="space-y-4">
            <details className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
              <summary className="font-medium cursor-pointer">What Python versions are supported?</summary>
              <p className="mt-2 text-gray-600 dark:text-gray-400">
                pHdockUI supports Python 3.8 and above. We recommend using Python 3.9 or 3.10 for optimal performance.
              </p>
            </details>

            <details className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
              <summary className="font-medium cursor-pointer">How accurate are the pKa predictions?</summary>
              <p className="mt-2 text-gray-600 dark:text-gray-400">
                Our ensemble model achieves ±0.5 pKa units RMSE on standard benchmarks, outperforming most commercial tools.
              </p>
            </details>

            <details className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
              <summary className="font-medium cursor-pointer">Can I use my own docking backend?</summary>
              <p className="mt-2 text-gray-600 dark:text-gray-400">
                Yes! pHdockUI supports AutoDock Vina, Glide, and custom backends through our plugin interface.
              </p>
            </details>
          </div>
        </section>
      </div>
    </div>
  );
} 