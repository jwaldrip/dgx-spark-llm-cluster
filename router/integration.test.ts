import { test, expect, beforeAll, afterAll } from "bun:test";

// End-to-end over real HTTP: three mock replicas, the real router process, a growing
// conversation. Proves the wire path and the affinity headers, not just the pure functions.

type Server = { stop: () => void; port: number; hits: () => number };

function mockReplica(): Server {
  let hits = 0;
  const s = Bun.serve({
    port: 0,
    fetch(req) {
      const u = new URL(req.url);
      if (u.pathname === "/health") return new Response("ok");
      hits++;
      return Response.json({
        choices: [{ message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
    },
  });
  return { stop: () => s.stop(true), port: s.port, hits: () => hits };
}

let replicas: Server[];
let router: Bun.Subprocess;
const ROUTER_PORT = 18711;

beforeAll(async () => {
  replicas = [mockReplica(), mockReplica(), mockReplica()];
  router = Bun.spawn(["bun", "run", `${import.meta.dir}/prefix-affinity-router.ts`], {
    env: {
      ...process.env,
      ROUTER_PORT: String(ROUTER_PORT),
      ROUTER_UPSTREAMS: replicas.map((r) => `http://127.0.0.1:${r.port}`).join(","),
      ROUTER_MAX_INFLIGHT: "4",
      // Probe fast so a health transition is observable within a test, instead of waiting
      // out the 5s production interval.
      ROUTER_PROBE_MS: "150",
    },
    stdout: "pipe",
    stderr: "pipe",
  });
  for (let i = 0; i < 60; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${ROUTER_PORT}/health`);
      if (res.ok) return;
    } catch {}
    // Real delay, deliberately: the router is a separate OS process, so fake timers in this
    // process cannot advance its clock. We poll a real readiness signal rather than guess a
    // fixed duration, and return the moment it is observed.
    await Bun.sleep(100);
  }
  throw new Error("router did not become ready");
});

afterAll(() => {
  router?.kill();
  for (const r of replicas) r.stop();
});

const SYSTEM = { role: "system", content: "You are a coding agent. ".repeat(200) };

function body(turns: number) {
  const messages: Array<{ role: string; content: string }> = [SYSTEM, { role: "user", content: "start" }];
  for (let i = 0; i < turns; i++) {
    messages.push({ role: "assistant", content: `a${i}` });
    messages.push({ role: "user", content: `u${i}` });
  }
  return { model: "qwen3.8-flash-next", messages };
}

async function post(turns: number) {
  const res = await fetch(`http://127.0.0.1:${ROUTER_PORT}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body(turns)),
  });
  expect(res.status).toBe(200);
  await res.json();
  return {
    replica: res.headers.get("X-Router-Replica"),
    affinity: res.headers.get("X-Router-Affinity"),
  };
}

test("a growing conversation lands on one replica every turn, over real HTTP", async () => {
  const seen = new Set<string>();
  for (const turns of [0, 1, 2, 4, 8, 16, 32]) {
    const r = await post(turns);
    expect(r.affinity).toBe("hit");
    seen.add(r.replica!);
  }
  expect(seen.size).toBe(1);

  const stats = await (await fetch(`http://127.0.0.1:${ROUTER_PORT}/router/stats`)).json();
  expect(stats.affinityRate).toBe(1);
  expect(stats.totalRouted).toBe(7);

  // exactly one mock replica saw traffic
  expect(replicas.filter((r) => r.hits() > 0).length).toBe(1);
});

test("distinct system prompts reach more than one replica", async () => {
  const seen = new Set<string>();
  for (let i = 0; i < 40; i++) {
    const res = await fetch(`http://127.0.0.1:${ROUTER_PORT}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "m", messages: [{ role: "system", content: `variant ${i}` }] }),
    });
    await res.json();
    seen.add(res.headers.get("X-Router-Replica")!);
  }
  expect(seen.size).toBeGreaterThan(1);
});

test("router reports unhealthy when every replica is stopped", async () => {
  for (const r of replicas) r.stop();
  // wait for the probe interval to observe the loss
  for (let i = 0; i < 40; i++) {
    const res = await fetch(`http://127.0.0.1:${ROUTER_PORT}/health`);
    if (res.status === 503) return;
    // Real delay for the same reason: the health verdict flips on the child process's own
    // probe interval, which this process cannot drive. Polls the actual condition.
    await Bun.sleep(250);
  }
  throw new Error("router never reported 503 after losing all replicas");
});
