import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Remove 'standalone' output for Vercel deployment
  // output: 'standalone', // Only needed for Docker
  serverExternalPackages: [],
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
};

export default nextConfig;
