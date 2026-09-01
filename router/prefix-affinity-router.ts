#!/usr/bin/env bun
//
// Prefix-affinity router for a vLLM data-parallel replica set.
//
// Why this exists. vLLM's own DP load balancer routes by queue depth, which is correct for
// a stateless service and wrong for this one: every replica keeps its OWN prefix cache. A
// session whose turns scatter across replicas pays a cold prefill on each hop. Measured on
// this hardware, a 184K-token prefill costs 103.6 s cold and 1.82 s against a warm cache,
// and the workload runs 181 prompt tokens per generated token. Losing affinity is therefore
// the single most expensive mistake available.
//
// Strategy: hash the STABLE PREFIX of the request, not the whole body. The stable prefix is
// the system message plus the tool schemas plus the first user turn, which is exactly the
// span a growing conversation keeps re-sending unchanged. Identical prefix -> same replica,
// so that replica's cache stays warm across every turn of the session.
//
// Load is the tie-breaker, never the primary key: a replica is skipped only when it is
// saturated or unhealthy, and the fallback is deterministic (rendezvous order) so a
// transient spike does not permanently rehome a session.

const UPSTREAMS = (process.env.ROUTER_UPSTREAMS ?? "http://127.0.0.1:8000")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

const PORT = Number(process.env.ROUTER_PORT ?? 18010);
// A replica is "busy" past this many in-flight requests. Sized from KV, not guessed:
// KV pool tokens / observed mean prompt tokens. 519,943 / 122,835 = 4.23 -> 4.
const MAX_INFLIGHT = Number(process.env.ROUTER_MAX_INFLIGHT ?? 4);
const PROBE_MS = Number(process.env.ROUTER_PROBE_MS ?? 5000);

type Replica = {
  url: string;
  inflight: number;
  healthy: boolean;
  routed: number;
  affinityHits: number;
  displaced: number;
};

const replicas: Replica[] = UPSTREAMS.map((url) => ({
  url,
  inflight: 0,
  healthy: true,
  routed: 0,
  affinityHits: 0,
  displaced: 0,
}));

// ---------------------------------------------------------------- affinity key

type ChatMessage = { role: string; content?: unknown };

/** Narrow an unvalidated array element without casting: request bodies are external input. */
function isChatMessage(value: unknown): value is ChatMessage {
  if (typeof value !== "object" || value === null) return false;
  if (!("role" in value)) return false;
  return typeof value.role === "string";
}

/**
 * The span of a request that stays byte-identical as a conversation grows.
 *
 * Deliberately excludes later turns: including them would change the key on every turn,
 * which is the same as having no affinity at all. Deliberately includes tool schemas,
 * because they sit inside the rendered system block and any reordering already breaks the
 * upstream cache regardless of routing.
 */
export function stablePrefix(body: Record<string, unknown>): string {
  const parts: string[] = [];
  if (typeof body.model === "string") parts.push(`model:${body.model}`);

  const tools = body.tools;
  if (Array.isArray(tools)) parts.push(`tools:${JSON.stringify(tools)}`);

  const messages = body.messages;
  if (Array.isArray(messages)) {
    for (const m of messages) {
      if (isChatMessage(m) && m.role === "system") {
        parts.push(`system:${JSON.stringify(m.content)}`);
      }
    }
    const firstUser = messages.find((m) => isChatMessage(m) && m.role === "user");
    if (isChatMessage(firstUser)) parts.push(`user0:${JSON.stringify(firstUser.content)}`);
  } else if (typeof body.prompt === "string") {
    // completions endpoint: the leading span is the only stable thing available
    parts.push(`prompt0:${body.prompt.slice(0, 4096)}`);
  }

  return parts.join("\u0000");
}

function hash64(s: string): bigint {
  // FNV-1a, 64 bit. Cheap, stable across processes, no crypto needed for placement.
  let h = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (let i = 0; i < s.length; i++) {
    h = (h ^ BigInt(s.charCodeAt(i) & 0xff)) * prime & mask;
  }
  return h;
}

/**
 * Rendezvous (highest random weight) ordering over replicas for a key.
 *
 * Chosen over modulo so that removing or adding a replica rehomes only its own share of
 * sessions instead of reshuffling every one of them, which would cold-start the entire
 * fleet's cache on any topology change.
 */
export function rendezvousOrder(key: string, urls: string[]): string[] {
  return urls
    .map((url) => ({ url, w: hash64(`${key}\u0001${url}`) }))
    .sort((a, b) => (a.w === b.w ? (a.url < b.url ? -1 : 1) : a.w > b.w ? -1 : 1))
    .map((x) => x.url);
}

/** Pick a replica: affinity first, load only as a veto. Returns null if all are unhealthy. */
export function pick(key: string, pool: Replica[], maxInflight: number): { replica: Replica; affinity: boolean } | null {
  const order = rendezvousOrder(key, pool.map((r) => r.url));
  const byUrl: Record<string, Replica> = Object.fromEntries(pool.map((r) => [r.url, r]));

  const preferred = byUrl[order[0]];
  if (preferred.healthy && preferred.inflight < maxInflight) {
    return { replica: preferred, affinity: true };
  }

  // Preferred replica is saturated or down. Walk the deterministic order so the same
  // session always displaces to the same second choice, keeping ITS cache warm too.
  for (const url of order.slice(1)) {
    const r = byUrl[url];
    if (r.healthy && r.inflight < maxInflight) return { replica: r, affinity: false };
  }

  // Everything is saturated. Queue on the preferred replica rather than spraying: waiting
  // for a warm cache beats a cold prefill elsewhere when prefill is 99% of the cost.
  if (preferred.healthy) return { replica: preferred, affinity: true };

  const anyHealthy = pool.find((r) => r.healthy);
  return anyHealthy ? { replica: anyHealthy, affinity: false } : null;
}

// ---------------------------------------------------------------- health probing

async function probe(): Promise<void> {
  await Promise.all(
    replicas.map(async (r) => {
      try {
        const res = await fetch(`${r.url}/health`, {
          signal: AbortSignal.timeout(3000),
        });
        r.healthy = res.ok;
      } catch {
        r.healthy = false;
      }
    }),
  );
}

// ---------------------------------------------------------------- server

if (import.meta.main) {
  await probe();
  setInterval(probe, PROBE_MS);

  Bun.serve({
    port: PORT,
    idleTimeout: 255,
    async fetch(req) {
      const url = new URL(req.url);

      if (url.pathname === "/router/stats") {
        const total = replicas.reduce((n, r) => n + r.routed, 0);
        const hits = replicas.reduce((n, r) => n + r.affinityHits, 0);
        return Response.json({
          upstreams: replicas.map(({ url, inflight, healthy, routed, affinityHits, displaced }) => ({
            url, inflight, healthy, routed, affinityHits, displaced,
          })),
          totalRouted: total,
          affinityRate: total ? hits / total : null,
          maxInflight: MAX_INFLIGHT,
        });
      }

      // Health of the fleet, so callers can probe the router the way they probe a replica.
      if (url.pathname === "/health") {
        const up = replicas.filter((r) => r.healthy).length;
        return up > 0
          ? new Response("ok", { status: 200 })
          : new Response("no healthy replicas", { status: 503 });
      }

      let bodyText: string | undefined;
      let key = "";
      if (req.method === "POST") {
        bodyText = await req.text();
        try {
          key = stablePrefix(JSON.parse(bodyText));
        } catch {
          key = url.pathname; // unparseable body: still deterministic, just coarse
        }
      } else {
        key = url.pathname;
      }

      const chosen = pick(key, replicas, MAX_INFLIGHT);
      if (!chosen) return new Response("no healthy replicas", { status: 503 });
      const { replica, affinity } = chosen;

      replica.inflight++;
      replica.routed++;
      if (affinity) replica.affinityHits++;
      else replica.displaced++;

      try {
        const upstream = await fetch(replica.url + url.pathname + url.search, {
          method: req.method,
          headers: { "Content-Type": "application/json" },
          body: bodyText,
        });
        // Stream the response through: a 262K-context reply must not be buffered whole.
        return new Response(upstream.body, {
          status: upstream.status,
          headers: {
            "Content-Type": upstream.headers.get("Content-Type") ?? "application/json",
            "X-Router-Replica": replica.url,
            "X-Router-Affinity": affinity ? "hit" : "displaced",
          },
        });
      } catch (err) {
        replica.healthy = false;
        return new Response(`upstream failed: ${(err as Error).message}`, { status: 502 });
      } finally {
        replica.inflight--;
      }
    },
  });

  console.log(
    `prefix-affinity-router ready on :${PORT} -> ${UPSTREAMS.join(", ")} (maxInflight ${MAX_INFLIGHT})`,
  );
}
