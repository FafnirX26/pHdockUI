import { Users } from "lucide-react";

export default function TeamSection() {
  const teamMembers = [
    {
      name: "Ravindra Lakkireddi",
      role: "Head of training, scoring, and data; ablation & quantum integration",
      image: "/team/member1.jpg"
    },
    {
      name: "Gianluca Radice", 
      role: "Defined GCNN frameworks; led SMILES→web pipeline",
      image: "/team/member2.jpg"
    },
    {
      name: "Denis Motuzenko",
      role: "Conceptual & chemical lead; interpretation layer & quantum logic",
      image: "/team/member3.jpg"
    }
  ];

  return (
    <section className="h-screen flex items-center justify-center px-4 bg-transparent">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">Meet Our Team</h2>
          <p className="text-gray-700 dark:text-gray-100">
            Passionate researchers dedicated to advancing computational chemistry
          </p>
        </div>

        <div className="space-y-12">
          {teamMembers.map((member, index) => (
            <div 
              key={index} 
              className={`flex flex-col md:flex-row items-center gap-8 ${
                index % 2 === 1 ? 'md:flex-row-reverse' : ''
              }`}
            >
              {/* Photo */}
              <div className="flex-shrink-0">
                <div className="w-40 h-40 md:w-48 md:h-48 bg-gray-200 dark:bg-gray-700 rounded-full flex items-center justify-center shadow-md">
                  <Users className="text-gray-500 dark:text-gray-300" size={64} />
                </div>
              </div>
              
              {/* Info */}
              <div className={`text-center md:text-left ${index % 2 === 1 ? 'md:text-right' : ''}`}>
                <h3 className="text-3xl md:text-4xl font-extrabold mb-2 text-gray-900 dark:text-white">{member.name}</h3>
                <p className="text-2xl md:text-2xl text-blue-700 dark:text-blue-300 leading-snug">{member.role}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}