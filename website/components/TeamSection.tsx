import { Users } from "lucide-react";

export default function TeamSection() {
  const teamMembers = [
    {
      name: "Team Member 1",
      role: "Lead Developer",
      image: "/team/member1.jpg"
    },
    {
      name: "Team Member 2", 
      role: "ML Engineer",
      image: "/team/member2.jpg"
    },
    {
      name: "Team Member 3",
      role: "Research Scientist",
      image: "/team/member3.jpg"
    }
  ];

  return (
    <section className="py-20 px-4 bg-white dark:bg-gray-900">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold mb-4">Meet Our Team</h2>
          <p className="text-gray-600 dark:text-gray-400">
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
                <div className="w-32 h-32 bg-gray-200 dark:bg-gray-700 rounded-full flex items-center justify-center">
                  <Users className="text-gray-400" size={48} />
                </div>
              </div>
              
              {/* Info */}
              <div className={`text-center md:text-left ${index % 2 === 1 ? 'md:text-right' : ''}`}>
                <h3 className="text-2xl font-bold mb-2">{member.name}</h3>
                <p className="text-xl text-blue-600 dark:text-blue-400">{member.role}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}