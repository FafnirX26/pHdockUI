import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const searchParams = req.nextUrl.searchParams;
    const stateId = searchParams.get('state_id') || '0';

    const fullBackendUrl = `${BACKEND_URL}/api/jobs/${id}/ligand-pdb?state_id=${stateId}`;
    console.log(`Fetching ligand PDB from: ${fullBackendUrl}`); // DEBUGGING

    const response = await fetch(fullBackendUrl, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `Backend returned ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Unknown error";
    console.error("Get ligand PDB error:", e);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
