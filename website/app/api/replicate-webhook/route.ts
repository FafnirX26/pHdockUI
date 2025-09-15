import { NextRequest, NextResponse } from "next/server";
import crypto from "crypto";

function verifySignature(payload: string, signature: string, secret: string) {
  const hmac = crypto.createHmac("sha256", secret);
  hmac.update(payload);
  const digest = `sha256=${hmac.digest("hex")}`;
  return crypto.timingSafeEqual(Buffer.from(digest), Buffer.from(signature));
}

export async function POST(req: NextRequest) {
  try {
    const secret = process.env.REPLICATE_WEBHOOK_SIGNING_SECRET;
    if (!secret) return NextResponse.json({ ok: true }); // Skip verify if unset

    const raw = await req.text();
    const sig = req.headers.get("replicate-signature") || "";
    const ok = verifySignature(raw, sig, secret);
    if (!ok) return NextResponse.json({ error: "invalid signature" }, { status: 401 });

    // For now we just acknowledge; UI polls /api/jobs/:id
    return NextResponse.json({ ok: true });
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Unknown error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}

