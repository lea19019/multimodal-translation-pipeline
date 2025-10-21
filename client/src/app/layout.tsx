import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Multimodal Translation Client',
  description: 'Web client for multimodal translation services',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-sans">{children}</body>
    </html>
  );
}
