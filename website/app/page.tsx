import Hero from "@/components/Hero";
import TeamSection from "@/components/TeamSection";
import CredibilitySection from "@/components/CredibilitySection";
import MoleculeInterface from "@/components/MoleculeInterface";
import Reveal from "@/components/Reveal";

export default function Home() {
  return (
    <div className="min-h-screen">
      <Reveal once={false}>
        <Hero />
      </Reveal>
      <Reveal delayMs={80} once={false}>
        <TeamSection />
      </Reveal>
      <Reveal delayMs={120} once={false}>
        <CredibilitySection />
      </Reveal>
      <Reveal delayMs={160} once={false}>
        <MoleculeInterface />
      </Reveal>
    </div>
  );
}
