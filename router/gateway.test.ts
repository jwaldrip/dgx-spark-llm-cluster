import { test, expect, beforeAll, afterAll } from "bun:test";
import { requestedModel } from "./gateway.ts";

// One endpoint, two model families, replica sets of different sizes. Proves model routing
// and per-model affinity together over real HTTP.

type Mock = { stop: () => void; port: number; seen: () => string[] };

function mockReplica(modelId: string): Mock {
  const seen: string[] = [];
  const s = Bun.serve({
    port: 0,
    fetch(req) {
      const u = new URL(req.url);
      if (u.pathname === "/health") return new Response("ok");
      seen.push(u.pathname);
      return Response.json({
        model: modelId,
        choices: [{ message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
    },
  });
  return { stop: () => s.stop(true), port: s.port, seen: () => seen };
}

let qwen: Mock[];
let glm: Mock[];
let gw: Bun.Subprocess;
const PORT = 18712;
const BASE = `http://127.0.0.1:${PORT}`;

beforeAll(async () => {
  qwen = [mockReplica("qwen"), mockReplica("qwen"), mockReplica("qwen")];
  glm = [mockReplica("glm")];
  const models = {
    "qwen3.8-flash-next": qwen.map((m) => `http://127.0.0.1:${m.port}`),
    "glm-5.3-flash": glm.map((m) => `http://127.0.0.1:${m.port}`),
  };
  gw = Bun.spawn(["bun", "run", `${import.meta.dir}/gateway.ts`], {
    env: {
      ...process.env,
      ROUTER_PORT: String(PORT),
      ROUTER_MODELS: JSON.stringify(models),
      ROUTER_MAX_INFLIGHT: "4",
      // Probe fast so health transitions are observable inside a test rather than after the
      // 5s production interval.
      ROUTER_PROBE_MS: "150",
    },
    stdout: "pipe",
    stderr: "pipe",
  });
  for (let i = 0; i < 80; i++) {
    try {
      if ((await fetch(`${BASE}/health`)).ok) return;
    } catch {}
    // Real delay: the gateway is a separate OS process, so fake timers here cannot advance
    // its clock. Polls the actual readiness signal and returns as soon as it is observed.
    await Bun.sleep(100);
  }
  throw new Error("gateway did not become ready");
});

afterAll(() => {
  gw?.kill();
  for (const m of [...qwen, ...glm]) m.stop();
});

const SYSTEM = { role: "system", content: "You are a coding agent. ".repeat(200) };

function chat(model: string, turns: number) {
  const messages: Array<{ role: string; content: string }> = [SYSTEM, { role: "user", content: "start" }];
  for (let i = 0; i < turns; i++) {
    messages.push({ role: "assistant", content: `a${i}` });
    messages.push({ role: "user", content: `u${i}` });
  }
  return { model, messages };
}

async function post(model: string, turns = 0) {
  const res = await fetch(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(chat(model, turns)),
  });
  const json: unknown = await res.json();
  return {
    status: res.status,
    model: res.headers.get("X-Router-Model"),
    replica: res.headers.get("X-Router-Replica"),
    affinity: res.headers.get("X-Router-Affinity"),
    json,
  };
}

// --- the single-endpoint contract ---

test("one catalog lists every model the cluster serves", async () => {
  const res = await fetch(`${BASE}/v1/models`);
  expect(res.status).toBe(200);
  const body = (await res.json()) as { data: Array<{ id: string; replicas: number; healthy_replicas: number }> };
  const ids = body.data.map((m) => m.id).sort();
  expect(ids).toEqual(["glm-5.3-flash", "qwen3.8-flash-next"]);
  const qwenEntry = body.data.find((m) => m.id === "qwen3.8-flash-next")!;
  expect(qwenEntry.replicas).toBe(3);
  expect(qwenEntry.healthy_replicas).toBe(3);
});

test("a request reaches the replica set of the model it names", async () => {
  const a = await post("qwen3.8-flash-next");
  const b = await post("glm-5.3-flash");
  expect(a.model).toBe("qwen3.8-flash-next");
  expect(b.model).toBe("glm-5.3-flash");

  const qwenPorts = qwen.map((m) => `http://127.0.0.1:${m.port}`);
  const glmPorts = glm.map((m) => `http://127.0.0.1:${m.port}`);
  expect(qwenPorts).toContain(a.replica);
  expect(glmPorts).toContain(b.replica);
  // no cross-family leakage
  expect(glmPorts).not.toContain(a.replica);
  expect(qwenPorts).not.toContain(b.replica);
});

test("an unknown model fails loudly and names what is served", async () => {
  const res = await fetch(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "llama-9000", messages: [SYSTEM] }),
  });
  expect(res.status).toBe(404);
  const body = (await res.json()) as { error: { message: string; served_models: string[] } };
  expect(body.error.message).toContain("llama-9000");
  expect(body.error.served_models.sort()).toEqual(["glm-5.3-flash", "qwen3.8-flash-next"]);
});

test("a request naming no model is rejected when the cluster serves several", async () => {
  const res = await fetch(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: [SYSTEM] }),
  });
  expect(res.status).toBe(404);
  const body = (await res.json()) as { error: { param: string } };
  expect(body.error.param).toBe("model");
});

// --- affinity survives the extra routing layer ---

test("a growing conversation stays on one replica within its model", async () => {
  const seen = new Set<string>();
  for (const turns of [0, 1, 2, 4, 8, 16, 32]) {
    const r = await post("qwen3.8-flash-next", turns);
    expect(r.status).toBe(200);
    expect(r.affinity).toBe("hit");
    seen.add(r.replica!);
  }
  expect(seen.size).toBe(1);

  const stats = (await (await fetch(`${BASE}/router/stats`)).json()) as {
    models: Array<{ model: string; affinityRate: number | null; totalRouted: number }>;
  };
  const q = stats.models.find((m) => m.model === "qwen3.8-flash-next")!;
  expect(q.affinityRate).toBe(1);
  expect(q.totalRouted).toBeGreaterThanOrEqual(7);
});

test("two model families are load-isolated from each other", async () => {
  const before = glm[0].seen().length;
  for (let i = 0; i < 12; i++) await post("qwen3.8-flash-next", i);
  expect(glm[0].seen().length).toBe(before); // qwen traffic never touched the glm replica
});

test("losing one model's replicas leaves the other routable", async () => {
  for (const m of glm) m.stop();
  for (let i = 0; i < 60; i++) {
    const res = await fetch(`${BASE}/v1/models`);
    const body = (await res.json()) as { data: Array<{ id: string; healthy_replicas: number }> };
    const g = body.data.find((m) => m.id === "glm-5.3-flash")!;
    if (g.healthy_replicas === 0) {
      // the cluster is still healthy overall, and qwen still serves
      expect((await fetch(`${BASE}/health`)).status).toBe(200);
      const r = await post("qwen3.8-flash-next");
      expect(r.status).toBe(200);
      const dead = await post("glm-5.3-flash");
      expect(dead.status).toBe(503);
      return;
    }
    // Real delay: waiting on the child process's own probe cycle, which this process
    // cannot drive with fake timers.
    await Bun.sleep(100);
  }
  throw new Error("gateway never observed the glm replica loss");
});

// --- unit surface ---

test("requestedModel tolerates junk without throwing", () => {
  expect(requestedModel({ model: "x" })).toBe("x");
  expect(requestedModel({ model: 42 })).toBeUndefined();
  expect(requestedModel(null)).toBeUndefined();
  expect(requestedModel("nope")).toBeUndefined();
  expect(requestedModel({})).toBeUndefined();
});
