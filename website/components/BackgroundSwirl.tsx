"use client";

import Image from "next/image";

export default function BackgroundSwirl() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-50 overflow-hidden">
      <div className="absolute inset-0">
        <Image
          src="/gradient-swirl.png"
          alt=""
          fill
          priority
          sizes="100vw"
          className="opacity-70 blur-3xl object-cover"
          style={{ objectFit: "cover", transform: "rotate(90deg) scale(1.15)" }}
        />
      </div>
    </div>
  );
}

