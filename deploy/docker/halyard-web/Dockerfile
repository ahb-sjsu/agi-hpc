# syntax=docker/dockerfile:1.7

# ─────────────────────────────────────────────────────────────────
# halyard-web — Next.js production image
#
# Three stages:
#   deps      — install node_modules from a cold package*.json
#   builder   — compile the Next.js standalone bundle
#   runner    — minimal runtime, copies just the standalone output
#
# The standalone bundle carries only the deps that the server side
# actually imports at runtime, so the final image is small (~180 MB
# uncompressed) and has no dev deps.
# ─────────────────────────────────────────────────────────────────

ARG NODE_VERSION=20.18.1

# ─────────────────────────────── deps ───────────────────────────
FROM node:${NODE_VERSION}-alpine AS deps
WORKDIR /app
# libc6-compat fixes the occasional glibc-shaped binary in
# Next.js's SWC pipeline on alpine.
RUN apk add --no-cache libc6-compat
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund

# ─────────────────────────────── builder ────────────────────────
FROM node:${NODE_VERSION}-alpine AS builder
WORKDIR /app
RUN apk add --no-cache libc6-compat
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# NEXT_PUBLIC_* are baked into the client bundle at build time.
# Override with `--build-arg` in CI, or use a default that points
# at the production Caddy host.
ARG NEXT_PUBLIC_LIVEKIT_URL=wss://halyard.atlas-sjsu.duckdns.org
ARG NEXT_PUBLIC_STATE_HTTP=https://halyard.atlas-sjsu.duckdns.org/state
ARG NEXT_PUBLIC_TOKEN_MINT_URL=https://halyard.atlas-sjsu.duckdns.org/keeper/api/livekit/token
ENV NEXT_PUBLIC_LIVEKIT_URL=$NEXT_PUBLIC_LIVEKIT_URL
ENV NEXT_PUBLIC_STATE_HTTP=$NEXT_PUBLIC_STATE_HTTP
ENV NEXT_PUBLIC_TOKEN_MINT_URL=$NEXT_PUBLIC_TOKEN_MINT_URL

ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ─────────────────────────────── runner ─────────────────────────
FROM node:${NODE_VERSION}-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000
ENV PORT=3000
# Bind loopback: the container sits behind atlas-caddy in prod.
# Docker publishes the port to the host; the host binds that only
# to 127.0.0.1 via the `docker run -p 127.0.0.1:3000:3000` form.
ENV HOSTNAME=0.0.0.0

CMD ["node", "server.js"]
