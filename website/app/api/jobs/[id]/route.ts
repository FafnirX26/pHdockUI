import { NextRequest, NextResponse } from "next/server";

export async function GET(
  _req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const token = process.env.REPLICATE_API_TOKEN;
    if (!token) {
      return NextResponse.json({ error: "REPLICATE_API_TOKEN not set" }, { status: 500 });
    }

    const id = params.id;
    const res = await fetch(`https://api.replicate.com/v1/predictions/${id}`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    if (!res.ok) {
      const t = await res.text();
      return NextResponse.json({ error: t }, { status: res.status });
    }
    type ReplicatePrediction = {
      id: string;
      status: string;
      output?: unknown;
      error?: string;
      metrics?: { predict_time?: number };
    };
    const p: ReplicatePrediction = await res.json();

    // Normalize payload for UI: if completed with output, map to {status, progress, results}
    let out: Record<string, unknown> = { status: p.status, progress: p.metrics?.predict_time ? 1.0 : 0.0 };
    if (p.status === "succeeded" && typeof p.output !== "undefined") {
      // Our predictor returns an object already matching UI expectations
      return NextResponse.json(p.output);
    } else if (p.error) {
      out = { status: "failed", error: p.error, progress: 0 };
    }
    return NextResponse.json(out);
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Unknown error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}