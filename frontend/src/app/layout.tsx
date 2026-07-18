import type { Metadata } from "next";
import { Fraunces, Space_Grotesk } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space",
  subsets: ["latin"],
  display: "swap",
});

const fraunces = Fraunces({
  variable: "--font-fraunces",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL("https://movie-reccomender-system-red.vercel.app"),
  title: "Shortlist | Find your next movie",
  description:
    "Pick a few movies you love and get a personal shortlist from an ML model trained on MovieLens ratings.",
  alternates: { canonical: "/" },
  openGraph: {
    title: "Shortlist | Find your next movie",
    description: "Mix a few favorites and get a fresh movie shortlist built around your taste.",
    url: "/",
    siteName: "Shortlist",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Shortlist | Find your next movie",
    description: "Mix a few favorites and get a fresh movie shortlist built around your taste.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${spaceGrotesk.variable} ${fraunces.variable}`}>
      <body>{children}</body>
    </html>
  );
}
