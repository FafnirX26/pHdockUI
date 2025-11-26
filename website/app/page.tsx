import Hero from "@/components/Hero";
import TeamSection from "@/components/TeamSection";
import CredibilitySection from "@/components/CredibilitySection";
import Reveal from "@/components/Reveal";
import MoleculeInterfaceLoader from "@/components/MoleculeInterfaceLoader";

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
        <MoleculeInterfaceLoader />
      </Reveal>
    </div>
  );
}
