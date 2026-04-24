/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Produce a self-contained standalone bundle so the Dockerfile
  // can copy just `.next/standalone` + `.next/static` and not need
  // node_modules at runtime.
  output: "standalone",
  eslint: {
    // Lint via `npm run lint`, not as a blocker on `next build`.
    ignoreDuringBuilds: true,
  },
  // The LiveKit React SDK ships ESM-only modules; Next.js 14
  // handles them natively but we allow transpilation for older
  // consumers. Pinned list keeps the build deterministic.
  transpilePackages: [
    "@livekit/components-react",
    "@livekit/components-styles",
    "livekit-client",
  ],
};

export default nextConfig;
