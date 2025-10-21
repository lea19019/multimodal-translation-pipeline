'use client';

export default function DebugPage() {
  return (
    <div style={{ padding: '20px', fontFamily: 'monospace' }}>
      <h1>Environment Variables Debug</h1>
      <pre>
        NEXT_PUBLIC_GATEWAY_URL: {process.env.NEXT_PUBLIC_GATEWAY_URL || 'undefined'}
        {'\n'}
        NEXT_PUBLIC_ASR_URL: {process.env.NEXT_PUBLIC_ASR_URL || 'undefined'}
        {'\n'}
        NEXT_PUBLIC_NMT_URL: {process.env.NEXT_PUBLIC_NMT_URL || 'undefined'}
        {'\n'}
        NEXT_PUBLIC_TTS_URL: {process.env.NEXT_PUBLIC_TTS_URL || 'undefined'}
      </pre>
    </div>
  );
}
