import { ImageResponse } from "next/og";

export const alt = "Shortlist movie recommendations";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpenGraphImage() {
  return new ImageResponse(
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        padding: "64px 72px",
        color: "#f6efe5",
        background: "linear-gradient(135deg, #0d0b0a 0%, #241a13 62%, #5a351d 100%)",
        fontFamily: "sans-serif",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 18, fontSize: 30 }}>
        <div
          style={{
            width: 52,
            height: 52,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            borderRadius: 16,
            color: "#17110d",
            background: "#f1b65c",
            fontWeight: 800,
          }}
        >
          S
        </div>
        Shortlist
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <div style={{ fontSize: 76, lineHeight: 1, letterSpacing: -3, fontWeight: 650 }}>
          Find something worth watching.
        </div>
        <div style={{ fontSize: 30, color: "#c8bbae" }}>
          Pick a few favorites. Get a movie mix built around your taste.
        </div>
      </div>
      <div style={{ display: "flex", gap: 36, color: "#f1b65c", fontSize: 23 }}>
        <span>87,585 movies</span>
        <span>186,458 taste profiles</span>
        <span>One fresh shortlist</span>
      </div>
    </div>,
    size
  );
}
