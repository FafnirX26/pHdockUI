import Hero from "@/components/Hero";
import TeamSection from "@/components/TeamSection";
import CredibilitySection from "@/components/CredibilitySection";
import MoleculeInterface from "@/components/MoleculeInterface";

export default function Home() {
  return (
    <div className="min-h-screen">
      <Hero />
      <TeamSection />
      <CredibilitySection />
      <MoleculeInterface />
    </div>
  );
}
