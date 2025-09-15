import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const token = process.env.REPLICATE_API_TOKEN;
    const modelVersion = process.env.REPLICATE_MODEL_ID; // owner/model:version

    if (!token || !modelVersion) {
      return NextResponse.json(
        { error: "Server not configured. Set REPLICATE_API_TOKEN and REPLICATE_MODEL_ID." },
        { status: 500 }
      );
    }

    const body = await req.json();
    const { smiles, sdf_content, ph_value = 7.4, ensemble_size = 5 } = body || {};

    const webhookUrl = process.env.REPLICATE_WEBHOOK_URL;

    const createRes = await fetch("https://api.replicate.com/v1/predictions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        version: modelVersion,
        input: { smiles: smiles ?? null, sdf_content: sdf_content ?? null, ph_value, ensemble_size },
        ...(webhookUrl
          ? { webhook: webhookUrl, webhook_events_filter: ["completed"] }
          : {}),
      }),
    });

    if (!createRes.ok) {
      const err = await createRes.text();
      return NextResponse.json({ error: err }, { status: createRes.status });
    }

    const prediction = await createRes.json();
    return NextResponse.json({
      job_id: prediction.id,
      status: prediction.status,
      created_at: prediction.created_at,
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Unknown error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}

