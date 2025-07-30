import { Lightbulb, Target, Users } from "lucide-react";

export default function AboutPage() {
  const teamMembers = [
    {
      name: "Team Member 1",
      role: "Lead Developer",
      bio: "PhD in Computational Chemistry with expertise in molecular modeling and machine learning.",
      image: "/team/member1.jpg"
    },
    {
      name: "Team Member 2", 
      role: "ML Engineer",
      bio: "Specializes in graph neural networks and chemical property prediction.",
      image: "/team/member2.jpg"
    },
    {
      name: "Team Member 3",
      role: "Research Scientist",
      bio: "Background in drug discovery and molecular docking methodologies.",
      image: "/team/member3.jpg"
    }
  ];

  return (
    <div className="min-h-screen py-20 px-4">
      <div className="container mx-auto max-w-6xl">
        <h1 className="text-4xl font-bold text-center mb-12">About pHdockUI</h1>

        {/* Mission Section */}
        <section className="mb-16">
          <div className="max-w-3xl mx-auto text-center">
            <Target className="mx-auto mb-4 text-blue-600" size={48} />
            <h2 className="text-2xl font-semibold mb-4">Our Mission</h2>
            <p className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed">
              We&apos;re advancing computational drug discovery by accounting for pH-dependent 
              molecular behavior. Traditional docking tools often overlook protonation states, 
              leading to inaccurate predictions. Our suite bridges this gap with state-of-the-art 
              machine learning and quantum-informed models.
            </p>
          </div>
        </section>

        {/* Inspiration Section */}
        <section className="mb-16 bg-gray-50 dark:bg-gray-800 rounded-2xl p-8">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-center gap-3 mb-4">
              <Lightbulb className="text-yellow-500" size={32} />
              <h2 className="text-2xl font-semibold">Our Inspiration</h2>
            </div>
            <div className="space-y-4 text-gray-700 dark:text-gray-300">
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
            <p className="text-gray-600 dark:text-gray-400 mt-2">
              Passionate researchers dedicated to advancing computational chemistry
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {teamMembers.map((member, index) => (
              <div key={index} className="text-center">
                <div className="w-48 h-48 mx-auto mb-4 bg-gray-200 dark:bg-gray-700 rounded-full flex items-center justify-center">
                  <Users className="text-gray-400" size={64} />
                </div>
                <h3 className="text-xl font-semibold mb-1">{member.name}</h3>
                <p className="text-blue-600 dark:text-blue-400 mb-3">{member.role}</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">{member.bio}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Academic Affiliations */}
        <section className="mb-16">
          <h2 className="text-2xl font-semibold text-center mb-8">Academic Affiliations</h2>
          <div className="flex flex-wrap justify-center gap-8 items-center">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-400">Your University</div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Department of Chemistry</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-400">Research Institute</div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Computational Biology Lab</p>
            </div>
          </div>
        </section>

        {/* Acknowledgments */}
        <section className="bg-gray-50 dark:bg-gray-800 rounded-2xl p-8">
          <h2 className="text-2xl font-semibold mb-4">Acknowledgments</h2>
          <p className="text-gray-700 dark:text-gray-300">
            We thank our advisors, collaborators, and the open-source community for their 
            invaluable contributions. Special thanks to the developers of RDKit, PyTorch Geometric, 
            and AutoDock for providing the foundation upon which this work builds.
          </p>
          <p className="text-gray-700 dark:text-gray-300 mt-4">
            This research was supported by [Grant Agency] under award number [XXX].
          </p>
        </section>
      </div>
    </div>
  );
} 