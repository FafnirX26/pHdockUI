import { Lightbulb, Target, Users } from "lucide-react";

export default function AboutPage() {
  const teamMembers = [
    {
      name: "Ravindra Lakkireddy",
      role: "Lead Developer",
      bio: "Head of training, scoring, and data collection, handling abalation studies as well as quantum integration",
      image: "/team/member1.jpg"
    },
    {
      name: "Gianluca Radice", 
      role: "ML Engineer",
      bio: "Defined the frameworks for the GCNNs used, lead efforts for constructing a full pipeline from SMILE to website output",
      image: "/team/member2.jpg"
    },
    {
      name: "Denis Motuzenko",
      role: "Research Scientist",
      bio: "Conceptual and chemical leader, engaged with the interpretation layer and organized quantum logic",
      image: "/team/member3.jpg"
    }
  ];

  return (
    <div className="min-h-screen py-20 px-4 bg-transparent">
      <div className="container mx-auto max-w-6xl">
        <h1 className="text-4xl font-bold text-center mb-12 inline-block rounded-lg border border-black/10 dark:border-white/10 bg-white/40 dark:bg-black/30 backdrop-blur px-3 py-1">About pHdockUI</h1>

        {/* Mission Section */}
        <section className="mb-16">
          <div className="max-w-3xl mx-auto text-center">
            <Target className="mx-auto mb-4 text-blue-600" size={48} />
            <h2 className="text-2xl font-semibold mb-4 inline-block rounded border border-black/10 dark:border-white/10 bg-white/40 dark:bg-black/30 backdrop-blur px-2 py-0.5">Our Mission</h2>
            <p className="text-lg text-gray-800 dark:text-gray-100 leading-relaxed inline-block rounded border border-black/10 dark:border-white/10 bg-white/40 dark:bg-black/30 backdrop-blur px-2 py-0.5">
              We&apos;re advancing computational drug discovery by accounting for pH-dependent 
              molecular behavior. Traditional docking tools often overlook protonation states, 
              leading to inaccurate predictions. Our suite bridges this gap with state-of-the-art 
              machine learning and quantum-informed models.
            </p>
          </div>
        </section>

        {/* Inspiration Section */}
        <section className="mb-16 rounded-2xl p-8 border border-black/10 dark:border-white/10 bg-white/40 dark:bg-black/30 backdrop-blur">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-center gap-3 mb-4">
              <Lightbulb className="text-yellow-500" size={32} />
              <h2 className="text-2xl font-semibold">Our Inspiration</h2>
            </div>
            <div className="space-y-4 text-gray-800 dark:text-gray-100">
              <p>
                The project emerged from observing critical failures in drug discovery pipelines 
                where promising candidates failed due to incorrect protonation state modeling. 
                At physiological pH, many drug molecules exist in multiple protonation states, 
                each with different binding affinities.
              </p>
              <p>
                Inspired by recent advances in graph neural networks and the availability of 
                large-scale pKa datasets, we developed an integrated approach that combines:
              </p>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li>Fast, accurate pKa prediction using ensemble ML models</li>
                <li>Quantum mechanical validation for challenging cases</li>
                <li>Seamless integration with popular docking tools</li>
                <li>User-friendly interfaces for both researchers and educators</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Team Section */}
        <section className="mb-16">
          <div className="text-center mb-12">
            <Users className="mx-auto mb-4 text-purple-600" size={48} />
            <h2 className="text-2xl font-semibold">Meet Our Team</h2>
            <p className="text-gray-700 dark:text-gray-100 mt-2 inline-block rounded border border-black/10 dark:border-white/10 bg-white/40 dark:bg-black/30 backdrop-blur px-2 py-0.5">
              Passionate researchers dedicated to advancing computational chemistry
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {teamMembers.map((member, index) => (
              <div key={index} className="text-center">
                <div className="w-48 h-48 mx-auto mb-4 bg-gray-200 dark:bg-gray-700 rounded-full flex items-center justify-center">
                  <Users className="text-gray-400" size={64} />
                </div>
                <h3 className="text-2xl md:text-3xl font-extrabold mb-1 text-gray-900 dark:text-white">{member.name}</h3>
                <p className="text-lg md:text-xl text-blue-700 dark:text-blue-300 mb-3">{member.role}</p>
                <p className="text-base text-gray-900 dark:text-white leading-snug">{member.bio}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Academic Affiliations */}
        <section className="mb-16">
          <h2 className="text-2xl font-semibold text-center mb-8">Academic Affiliations</h2>
          <div className="flex flex-wrap justify-center gap-8 items-center">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-700 dark:text-gray-100 inline-block rounded border border-black/10 dark:border-white/10 bg-white/40 dark:bg-black/30 backdrop-blur px-2 py-0.5">Poolesville High School</div>
              <p className="text-sm text-gray-700 dark:text-gray-200">Science, Math, and Computer Science Magnet Program (SMCS)</p>
            </div>
          </div>
        </section>

        {/* Acknowledgments */}
        <section className="rounded-2xl p-8 border border-black/10 dark:border-white/10 bg-white/40 dark:bg-black/30 backdrop-blur">
          <h2 className="text-2xl font-semibold mb-4">Acknowledgments</h2>
          <p className="text-gray-800 dark:text-gray-100">
            We thank our advisors, collaborators, and the open-source community for their 
            invaluable contributions. Special thanks to the developers of RDKit, PyTorch Geometric, 
            and AutoDock for providing the foundation upon which this work builds.
          </p>
        </section>
      </div>
    </div>
  );
} 