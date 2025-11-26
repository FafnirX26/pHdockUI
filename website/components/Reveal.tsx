"use client";

import React, { useEffect, useRef, useState } from "react";

type RevealProps = {
  children: React.ReactNode;
  delayMs?: number;
  yOffsetPx?: number;
  as?: React.ElementType;
  className?: string;
  once?: boolean;
};

export default function Reveal({
  children,
  delayMs = 0,
  yOffsetPx = 16,
  as: Tag = "div",
  className = "",
  once = true,
}: RevealProps) {
  const ref = useRef<HTMLElement | null>(null);
  const [visible, setVisible] = useState(false);
  const [isMotionReduced, setIsMotionReduced] = useState(false);

  useEffect(() => {
    // This check runs only on the client, after the component has mounted.
    setIsMotionReduced(
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    );

    const element = ref.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          if (once) observer.disconnect();
        } else if (!once) {
          setVisible(false);
        }
      },
      { threshold: 0.15 }
    );

    observer.observe(element);
    return () => observer.disconnect();
  }, [once]);

  const transitionStyle = isMotionReduced
    ? undefined
    : ({
        transition: "opacity 600ms ease, transform 600ms cubic-bezier(0.22, 1, 0.36, 1)",
        transitionDelay: `${delayMs}ms`,
      } as React.CSSProperties);

  return (
    <Tag
      ref={ref as React.RefObject<HTMLElement>}
      className={className}
      style={{
        opacity: visible || isMotionReduced ? 1 : 0,
        transform:
          visible || isMotionReduced ? "none" : `translateY(${yOffsetPx}px)`,
        willChange: isMotionReduced ? undefined : ("opacity, transform" as React.CSSProperties["willChange"]),
        ...transitionStyle,
      }}
    >
      {children}
    </Tag>
  );
}

