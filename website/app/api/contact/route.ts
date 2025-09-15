import { NextRequest, NextResponse } from "next/server";
import nodemailer from "nodemailer";

export async function POST(req: NextRequest) {
  try {
    const { name, email, institution, subject, message } = await req.json();

    const host = process.env.SMTP_HOST || "smtp.gmail.com";
    const port = Number(process.env.SMTP_PORT || 587);
    const user = process.env.SMTP_USER || "";
    const pass = process.env.SMTP_PASS || "";
    const from = process.env.SMTP_FROM || user;
    const to = process.env.SMTP_TO || "phdockteam@gmail.com";

    if (!user || !pass || !from) {
      return NextResponse.json({ error: "SMTP not configured" }, { status: 500 });
    }

    const transporter = nodemailer.createTransport({
      host,
      port,
      secure: port === 465,
      auth: { user, pass },
    });

    await transporter.sendMail({
      from,
      to,
      subject: `[pHdockUI Contact] ${subject}`,
      replyTo: email,
      text: `Name: ${name}\nEmail: ${email}\nInstitution: ${institution || "-"}\n\n${message}`,
    });

    return NextResponse.json({ status: "sent" });
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Unknown error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}

