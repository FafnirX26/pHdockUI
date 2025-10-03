import { Terminal, Download, Book, Github, ExternalLink } from "lucide-react";

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
                <code>{`# Clone repository
git clone https://github.com/FafnirX26/pHdock.git
cd pHdock

# Create virtual environment
python -m venv phd
source phd/bin/activate  # On Windows: phd\\Scripts\\activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn rdkit torch`}</code>
              </pre>
            </div>

            <div>
              <h3 className="font-medium mb-2">Basic Usage - Quick Start</h3>
              <pre className="bg-gray-900 dark:bg-gray-950 text-gray-100 p-4 rounded overflow-x-auto">
                <code>{`# Use the best quantum-enhanced model
python fast_quantum_pka.py

# Generate comprehensive performance summary
python final_summary.py`}</code>
              </pre>
            </div>

            <div>
              <h3 className="font-medium mb-2">Command Line Interface</h3>
              <pre className="bg-gray-900 dark:bg-gray-950 text-gray-100 p-4 rounded overflow-x-auto">
                <code>{`# Basic pKa prediction
python main.py --input molecules.smi --mode pka_prediction --output results/

# Use large dataset (17K+ molecules)
python main.py --input molecules.smi --data_source large --data_limit 10000

# Generate protonation states with optimized pH steps
python main.py --input molecules.sdf --mode protonation_states --ph_min 1 --ph_max 14

# Full pipeline with docking
python main.py --input molecules.smi --mode full_pipeline --receptor receptor.pdb --docking_tool gnina`}</code>
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
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6 space-y-6">
            <div className="border-l-4 border-blue-500 pl-4">
              <h3 className="font-mono font-semibold">Advanced ML Models</h3>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                State-of-the-art pKa prediction models
              </p>
              <div className="mt-4 space-y-2">
                <h4 className="font-medium">Available Models:</h4>
                <ul className="space-y-1 text-sm">
                  <li><code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">fast_quantum_pka.py</code> - Quantum-Enhanced Ensemble (R² = 0.874) - Best performing</li>
                  <li><code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">physics_informed_pka.py</code> - Physics-Informed Neural Network</li>
                  <li><code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">graph_neural_network_pka.py</code> - Graph Neural Network with attention</li>
                  <li><code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">ensemble_train_advanced.py</code> - Advanced ensemble with 47 features</li>
                </ul>
              </div>
            </div>

            <div className="border-l-4 border-purple-500 pl-4">
              <h3 className="font-mono font-semibold">Data Pipeline</h3>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                Comprehensive data processing with 17,000+ molecules from ChEMBL, IUPAC, SAMPL6
              </p>
              <div className="mt-4 space-y-2">
                <ul className="space-y-1 text-sm">
                  <li><code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">batch_data_fetcher.py</code> - Large-scale data acquisition</li>
                  <li><code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">data_filter.py</code> - Advanced filtering with statistical outlier detection</li>
                </ul>
              </div>
            </div>

            <div className="border-l-4 border-green-500 pl-4">
              <h3 className="font-mono font-semibold">Production Features</h3>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                pH-aware chemistry and docking integration
              </p>
              <div className="mt-4 space-y-2">
                <ul className="space-y-1 text-sm">
                  <li>Protonation states from pH 1-14 with variable step optimization</li>
                  <li>GNINA docking support with extensible framework</li>
                  <li>Saved models in models/ directory for immediate deployment</li>
                </ul>
              </div>
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
              href="https://github.com/FafnirX26/pHdock"
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
              <Download size={24} />
              <div>
                <h3 className="font-medium">Test Dataset</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">Sample molecules</p>
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
                Our quantum-enhanced ensemble model achieves R² = 0.874 on our test set of 17,000+ molecules, with performance comparable to expensive DFT calculations while maintaining computational efficiency suitable for high-throughput screening.
              </p>
            </details>

            <details className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
              <summary className="font-medium cursor-pointer">Can I use my own docking backend?</summary>
              <p className="mt-2 text-gray-600 dark:text-gray-400">
                Yes! pHdock currently supports GNINA with an extensible framework for additional docking tools. You can specify the docking tool using --docking_tool parameter.
              </p>
            </details>
          </div>
        </section>
      </div>
    </div>
  );
} 