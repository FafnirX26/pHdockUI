"use client";

import dynamic from 'next/dynamic';

const MoleculeInterface = dynamic(() => import('@/components/MoleculeInterface'), {
  ssr: false,
  loading: () => <div className="h-96 flex items-center justify-center"><p>Loading interface...</p></div>,
});

export default function MoleculeInterfaceLoader() {
  return <MoleculeInterface />;
}
