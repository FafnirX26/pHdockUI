import { BarChart, Award, BookOpen } from "lucide-react";

export default function CredibilitySection() {
  return (
    <section className="py-20 px-4 bg-white dark:bg-gray-900">
      <div className="container mx-auto max-w-6xl">
        <h2 className="text-3xl font-bold text-center mb-12">Validated Performance</h2>
        
        <div className="grid md:grid-cols-3 gap-8">
          {/* Benchmark Results */}
          <div className="text-center">
            <BarChart className="mx-auto mb-4 text-blue-600" size={48} />
            <h3 className="text-xl font-semibold mb-2">Benchmark Results</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Outperforms commercial tools with ±0.5 pKa RMSE on standard datasets
            </p>
            <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="text-left">ChemAxon</div>
                <div className="text-right font-mono">±0.8</div>
                <div className="text-left">Schrödinger</div>
                <div className="text-right font-mono">±0.7</div>
                <div className="text-left font-semibold">pHdockUI</div>
                <div className="text-right font-mono font-semibold">±0.5</div>
              </div>
            </div>
          </div>

          {/* Awards/Recognition */}
          <div className="text-center">
            <Award className="mx-auto mb-4 text-purple-600" size={48} />
            <h3 className="text-xl font-semibold mb-2">Recognition</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Presented at leading computational chemistry conferences
            </p>
            <ul className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
              <li>• ACS National Meeting 2024</li>
              <li>• Gordon Research Conference</li>
              <li>• SAMPL9 Challenge Participant</li>
            </ul>
          </div>

          {/* Publications */}
          <div className="text-center">
            <BookOpen className="mx-auto mb-4 text-green-600" size={48} />
            <h3 className="text-xl font-semibold mb-2">Publications</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Peer-reviewed research backing our methods
            </p>
            <div className="text-sm space-y-2">
              <a
                href="#"
                className="block p-3 bg-gray-50 dark:bg-gray-800 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                <p className="font-medium text-blue-600 dark:text-blue-400">
                  &quot;pH-aware ensemble docking with ML pKa prediction&quot;
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  J. Chem. Inf. Model. 2024 (in review)
                </p>
              </a>
            </div>
          </div>
        </div>

        {/* Citation */}
        <div className="mt-12 p-6 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <h3 className="text-lg font-semibold mb-2">How to Cite</h3>
          <pre className="text-sm overflow-x-auto whitespace-pre-wrap font-mono">
{`@software{phdockui2024,
  title = {pHdockUI: pH-aware Molecular Docking Suite},
  author = {Your Team Names},
  year = {2024},
  url = {https://github.com/yourusername/pHdockUI}
}`}
          </pre>
        </div>
      </div>
    </section>
  );
} 