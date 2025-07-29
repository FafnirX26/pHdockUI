import Hero from "@/components/Hero";
import MoleculeInterface from "@/components/MoleculeInterface";
import CredibilitySection from "@/components/CredibilitySection";

export default function Home() {
  return (
    <div className="min-h-screen">
      <Hero />
      <MoleculeInterface />
      <CredibilitySection />
    </div>
  );
}
