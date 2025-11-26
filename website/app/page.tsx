import Hero from "@/components/Hero";
import TeamSection from "@/components/TeamSection";
import CredibilitySection from "@/components/CredibilitySection";
import Reveal from "@/components/Reveal";
import dynamic from 'next/dynamic';

const MoleculeInterface = dynamic(() => import('@/components/MoleculeInterface'), {
  ssr: false,
  loading: () => <div className="h-96 flex items-center justify-center"><p>Loading interface...</p></div>,
});

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
