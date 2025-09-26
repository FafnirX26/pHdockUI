import { NextRequest, NextResponse } from "next/server";
import Replicate from "replicate";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const token = process.env.REPLICATE_API_TOKEN;
    if (!token) {
      return NextResponse.json({ error: "REPLICATE_API_TOKEN not set" }, { status: 500 });
    }

    // Initialize Replicate client
    const replicate = new Replicate({
      auth: token,
    });

    const { id } = await params;
    
    // Get prediction using Replicate library
    const prediction = await replicate.predictions.get(id);

    // Normalize payload for UI: if completed with output, map to {status, progress, results}
    let out: Record<string, unknown> = { 
      status: prediction.status, 
      progress: prediction.metrics?.predict_time ? 1.0 : 0.0 
    };
    
    if (prediction.status === "succeeded" && typeof prediction.output !== "undefined") {
      // Our predictor returns an object already matching UI expectations
      return NextResponse.json(prediction.output);
    } else if (prediction.error) {
      out = { status: "failed", error: prediction.error, progress: 0 };
    }
    
    return NextResponse.json(out);
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Unknown error";
    console.error("Replicate get prediction error:", e);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}