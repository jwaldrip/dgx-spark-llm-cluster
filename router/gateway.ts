#!/usr/bin/env bun
//
// One endpoint for the whole cluster. Many models behind it, each with its own replica set.
//
// Callers see a single OpenAI-compatible base URL whose /v1/models lists everything the
// cluster serves. Requests route by their `model` field to that model's replicas, then by
// prefix affinity within that set. Adding or moving a model is a config change here, not a
// new endpoint and not a client change.
//
// Two layers, in this order, because they solve different problems:
//
//   1. MODEL ROUTING. A request naming glm-5.3-flash must not land on a Qwen replica.
//      Unknown model names fail loudly with the list of known ones rather than silently
//      hitting whatever is first.
//
//   2. PREFIX AFFINITY within the model's replica set. Every vLLM replica keeps its OWN
//      prefix cache. Measured here: 87.7% hit rate over 22.4M queries on one node, and a
//      184K prefill costs 103.6 s cold against 1.82 s warm, on a workload running 181
//      prompt tokens per generated token. Load-based routing across N replicas throws most
//      of that away, so placement is keyed on the span of the request that stays identical
//      as a conversation grows.
//
// Config: ROUTER_MODELS as JSON, model id -> replica base URLs.
//   ROUTER_MODELS='{"qwen3.8-flash-next":["http://10.100.138.2:8000","http://10.100.138.1:8000"]}'

import { pick, stablePrefix, type Replica } from "./prefix-affinity-router.ts";

type ModelRoute = { model: string; replicas: Replica[]; maxInflight: number };

const PORT = Number(process.env.ROUTER_PORT ?? 18010);
const PROBE_MS = Number(process.env.ROUTER_PROBE_MS ?? 5000);
// Default admission per replica is derived, not chosen: KV pool tokens divided by the
// observed mean prompt length. 519,943 / 122,835 = 4.23 -> 4. Override per model below.
const DEFAULT_MAX_INFLIGHT = Number(process.env.ROUTER_MAX_INFLIGHT ?? 4);

function parseRoutes(raw: string): Record<string, ModelRoute> {
  const parsed: unknown = JSON.parse(raw);
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("ROUTER_MODELS must be a JSON object of model -> [urls]");
  }
  const routes: Record<string, ModelRoute> = {};
  for (const [model, value] of Object.entries(parsed)) {
    const urls = Array.isArray(value) ? value.filter((u): u is string => typeof u === "string") : [];
    if (urls.length === 0) throw new Error(`model ${model} has no replica URLs`);
    routes[model] = {
      model,
      maxInflight: DEFAULT_MAX_INFLIGHT,
      replicas: urls.map((url) => ({
        url, inflight: 0, healthy: true, routed: 0, affinityHits: 0, displaced: 0,
      })),
    };
  }
  return routes;
}

const ROUTES = parseRoutes(process.env.ROUTER_MODELS ?? '{"qwen3.8-flash-next":["http://127.0.0.1:8000"]}');

/** Read the target model from a request body without trusting its shape. */
export function requestedModel(body: unknown): string | undefined {
  if (typeof body !== "object" || body === null) return undefined;
  if (!("model" in body)) return undefined;
  return typeof body.model === "string" ? body.model : undefined;
}

async function probeAll(): Promise<void> {
  await Promise.all(
    Object.values(ROUTES).flatMap((route) =>
      route.replicas.map(async (r) => {
        try {
          const res = await fetch(`${r.url}/health`, { signal: AbortSignal.timeout(3000) });
          r.healthy = res.ok;
        } catch {
          r.healthy = false;
        }
      }),
    ),
  );
}

if (import.meta.main) {
  await probeAll();
  setInterval(probeAll, PROBE_MS);

  Bun.serve({
    port: PORT,
    idleTimeout: 255,
    async fetch(req) {
      const url = new URL(req.url);

      // One catalog for the whole cluster, so a client configures one provider.
      if (url.pathname === "/v1/models" && req.method === "GET") {
        return Response.json({
          object: "list",
          data: Object.values(ROUTES).map((route) => ({
            id: route.model,
            object: "model",
            owned_by: "cluster",
            // surfaced so a caller can see capacity without scraping metrics
            replicas: route.replicas.length,
            healthy_replicas: route.replicas.filter((r) => r.healthy).length,
          })),
        });
      }

      if (url.pathname === "/health") {
        // Healthy when at least one model is fully routable. A partially down cluster is
        // still useful, and reporting 503 for that would take everything offline.
        const anyModelUp = Object.values(ROUTES).some((r) => r.replicas.some((x) => x.healthy));
        return anyModelUp
          ? new Response("ok", { status: 200 })
          : new Response("no healthy replicas for any model", { status: 503 });
      }

      if (url.pathname === "/router/stats") {
        return Response.json({
          models: Object.values(ROUTES).map((route) => {
            const routed = route.replicas.reduce((n, r) => n + r.routed, 0);
            const hits = route.replicas.reduce((n, r) => n + r.affinityHits, 0);
            return {
              model: route.model,
              maxInflight: route.maxInflight,
              affinityRate: routed ? hits / routed : null,
              totalRouted: routed,
              replicas: route.replicas.map(({ url, inflight, healthy, routed, affinityHits, displaced }) => ({
                url, inflight, healthy, routed, affinityHits, displaced,
              })),
            };
          }),
        });
      }

      let bodyText: string | undefined;
      let model: string | undefined;
      let key = url.pathname;

      if (req.method === "POST") {
        bodyText = await req.text();
        try {
          const parsed: unknown = JSON.parse(bodyText);
          model = requestedModel(parsed);
          if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
            key = stablePrefix(parsed as Record<string, unknown>);
          }
        } catch {
          // leave key as the path: still deterministic, just coarse
        }
      }

      // Single-model clusters may omit the field; anything else must name a served model.
      const names = Object.keys(ROUTES);
      const target = model ?? (names.length === 1 ? names[0] : undefined);
      if (!target || !(target in ROUTES)) {
        return Response.json(
          {
            error: {
              message: target
                ? `model '${target}' is not served by this cluster`
                : "request did not name a model and this cluster serves more than one",
              type: "invalid_request_error",
              param: "model",
              served_models: names,
            },
          },
          { status: 404 },
        );
      }

      const route = ROUTES[target];
      const chosen = pick(key, route.replicas, route.maxInflight);
      if (!chosen) {
        return Response.json(
          { error: { message: `no healthy replica for model '${target}'`, type: "service_unavailable" } },
          { status: 503 },
        );
      }
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
        // Streamed through: a 262K-context response must not be buffered whole.
        return new Response(upstream.body, {
          status: upstream.status,
          headers: {
            "Content-Type": upstream.headers.get("Content-Type") ?? "application/json",
            "X-Router-Model": target,
            "X-Router-Replica": replica.url,
            "X-Router-Affinity": affinity ? "hit" : "displaced",
          },
        });
      } catch (err) {
        replica.healthy = false;
        return Response.json(
          { error: { message: `upstream failed: ${(err as Error).message}`, type: "bad_gateway" } },
          { status: 502 },
        );
      } finally {
        replica.inflight--;
      }
    },
  });

  const summary = Object.values(ROUTES)
    .map((r) => `${r.model} x${r.replicas.length}`)
    .join(", ");
  console.log(`cluster-gateway ready on :${PORT} serving ${summary}`);
}
