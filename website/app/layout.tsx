import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";
import BackgroundSwirl from "@/components/BackgroundSwirl";
import { Providers } from "@/components/Providers";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "pH-aware Molecular Docking Suite",
  description: "Advanced computational toolkit for pH-dependent molecular docking and pKa prediction",
  keywords: ["molecular docking", "pKa prediction", "drug discovery", "computational chemistry"],
  authors: [{ name: "pHdockUI Research Team" }],
  openGraph: {
    title: "pH-aware Molecular Docking Suite",
    description: "Advanced computational toolkit for pH-dependent molecular docking",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${inter.variable} font-sans antialiased bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100`}
      >
        <Providers>
          <div className="min-h-screen flex flex-col relative">
            <BackgroundSwirl />
            <Navigation />
            <main className="flex-1">
              {children}
            </main>
            <footer className="border-t border-gray-200 dark:border-gray-800 py-6 px-4">
              <div className="container mx-auto text-center text-sm text-gray-600 dark:text-gray-400">
                © 2024 pHdockUI Research Team. All rights reserved.
              </div>
            </footer>
          </div>
        </Providers>
      </body>
    </html>
  );
}
