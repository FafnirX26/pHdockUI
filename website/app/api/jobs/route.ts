import { NextRequest, NextResponse } from "next/server";
import Replicate from "replicate";

export async function POST(req: NextRequest) {
  try {
    const token = process.env.REPLICATE_API_TOKEN;
    const modelId = process.env.REPLICATE_MODEL_ID; // owner/model format

    if (!token || !modelId) {
      return NextResponse.json(
        { error: "Server not configured. Set REPLICATE_API_TOKEN and REPLICATE_MODEL_ID." },
        { status: 500 }
      );
    }

    // Initialize Replicate client
    const replicate = new Replicate({
      auth: token,
    });

    const body = await req.json();
    const { smiles, sdf_content, ph_value = 7.4, ensemble_size = 5 } = body || {};

    const webhookUrl = process.env.REPLICATE_WEBHOOK_URL;

    // Create prediction using Replicate library
    const input = { 
      smiles: smiles ?? null, 
      sdf_content: sdf_content ?? null, 
      ph_value, 
      ensemble_size 
    };

    const prediction = await replicate.predictions.create({
      model: modelId,
      input,
      ...(webhookUrl ? { 
        webhook: webhookUrl, 
        webhook_events_filter: ["completed"] 
      } : {})
    });

    return NextResponse.json({
      job_id: prediction.id,
      status: prediction.status,
      created_at: prediction.created_at,
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Unknown error";
    console.error("Replicate prediction error:", e);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}

