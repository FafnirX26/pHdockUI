import dynamic from 'next/dynamic';

const Hero = dynamic(() => import('@/components/Hero'), { ssr: false });
const TeamSection = dynamic(() => import('@/components/TeamSection'), { ssr: false });
const CredibilitySection = dynamic(() => import('@/components/CredibilitySection'), { ssr: false });
const Reveal = dynamic(() => import('@/components/Reveal'), { ssr: false });
const MoleculeInterfaceLoader = dynamic(() => import('@/components/MoleculeInterfaceLoader'), { ssr: false });

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
