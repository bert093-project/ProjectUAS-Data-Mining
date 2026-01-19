import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "UAS Data Mining Naive Bayes",
  description: "Syahril Karunia Pratama",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
