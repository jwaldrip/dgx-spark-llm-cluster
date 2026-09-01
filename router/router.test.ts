import { test, expect } from "bun:test";
import { stablePrefix, rendezvousOrder, pick } from "./prefix-affinity-router.ts";

const URLS = ["http://a:8000", "http://b:8000", "http://c:8000"];
const pool = (inflight: number[] = [0, 0, 0], healthy: boolean[] = [true, true, true]) =>
  URLS.map((url, i) => ({
    url,
    inflight: inflight[i],
    healthy: healthy[i],
    routed: 0,
    affinityHits: 0,
    displaced: 0,
  }));

const SYSTEM = { role: "system", content: "You are a coding agent. ".repeat(200) };
const TOOLS = [{ type: "function", function: { name: "read", parameters: {} } }];

/** A conversation that has grown by `turns` assistant/user pairs past the first user turn. */
function conversation(turns: number) {
  const messages: Array<{ role: string; content: string }> = [
    SYSTEM,
    { role: "user", content: "fix the failing test" },
  ];
  for (let i = 0; i < turns; i++) {
    messages.push({ role: "assistant", content: `step ${i}` });
    messages.push({ role: "user", content: `continue ${i}` });
  }
  return { model: "qwen3.8-flash-next", tools: TOOLS, messages };
}

// --- the property the whole design rests on ---

test("stable prefix is invariant as a conversation grows", () => {
  const first = stablePrefix(conversation(0));
  for (const turns of [1, 5, 20, 100]) {
    expect(stablePrefix(conversation(turns))).toBe(first);
  }
});

test("a session stays pinned to one replica across every turn", () => {
  const p = pool();
  const chosen = new Set<string>();
  for (const turns of [0, 1, 2, 5, 13, 40, 99]) {
    const key = stablePrefix(conversation(turns));
    chosen.add(pick(key, p, 4)!.replica.url);
  }
  expect(chosen.size).toBe(1);
});

test("different system prompts spread across replicas", () => {
  const p = pool();
  const hit: Record<string, number> = { "http://a:8000": 0, "http://b:8000": 0, "http://c:8000": 0 };
  for (let i = 0; i < 300; i++) {
    const key = stablePrefix({
      model: "m",
      messages: [{ role: "system", content: `agent variant ${i}` }, { role: "user", content: "go" }],
    });
    hit[pick(key, p, 4)!.replica.url]++;
  }
  // every replica used, and none starved: rendezvous should be roughly even
  for (const url of URLS) expect(hit[url]).toBeGreaterThan(300 / URLS.length / 2);
  expect(hit["http://a:8000"] + hit["http://b:8000"] + hit["http://c:8000"]).toBe(300);
});

// --- load only vetoes, never leads ---

test("a saturated preferred replica displaces deterministically, not randomly", () => {
  const key = stablePrefix(conversation(3));
  const preferred = pick(key, pool(), 4)!.replica.url;
  const idx = URLS.indexOf(preferred);

  const saturated = [0, 0, 0];
  saturated[idx] = 4;
  const first = pick(key, pool(saturated), 4)!;
  const second = pick(key, pool(saturated), 4)!;

  expect(first.affinity).toBe(false);
  expect(first.replica.url).not.toBe(preferred);
  expect(second.replica.url).toBe(first.replica.url); // same displacement every time
});

test("load below the cap never moves a session off its replica", () => {
  const key = stablePrefix(conversation(3));
  const preferred = pick(key, pool(), 4)!.replica.url;
  const idx = URLS.indexOf(preferred);
  const busy = [0, 0, 0];
  busy[idx] = 3; // busy but under the cap
  const got = pick(key, pool(busy), 4)!;
  expect(got.replica.url).toBe(preferred);
  expect(got.affinity).toBe(true);
});

test("when every replica is saturated it queues on the warm one", () => {
  const key = stablePrefix(conversation(3));
  const preferred = pick(key, pool(), 4)!.replica.url;
  const got = pick(key, pool([9, 9, 9]), 4)!;
  expect(got.replica.url).toBe(preferred);
  expect(got.affinity).toBe(true);
});

test("unhealthy replicas are skipped, and total loss reports null", () => {
  const key = stablePrefix(conversation(3));
  const preferred = pick(key, pool(), 4)!.replica.url;
  const idx = URLS.indexOf(preferred);
  const health = [true, true, true];
  health[idx] = false;
  const got = pick(key, pool([0, 0, 0], health))!;
  expect(got.replica.url).not.toBe(preferred);
  expect(got.replica.healthy).toBe(true);
  expect(pick(key, pool([0, 0, 0], [false, false, false]), 4)).toBeNull();
});

// --- topology changes must not cold-start the whole fleet ---

test("removing a replica rehomes only its own share of sessions", () => {
  const keys = Array.from({ length: 600 }, (_, i) =>
    stablePrefix({ model: "m", messages: [{ role: "system", content: `s${i}` }, { role: "user", content: "u" }] }),
  );
  const before = keys.map((k) => rendezvousOrder(k, URLS)[0]);
  const after = keys.map((k) => rendezvousOrder(k, URLS.slice(0, 2))[0]);

  let moved = 0;
  for (let i = 0; i < keys.length; i++) if (before[i] !== after[i]) moved++;

  // only sessions homed on the removed replica may move: about 1/3, never all of them
  expect(moved).toBeGreaterThan(0);
  expect(moved).toBeLessThan(keys.length * 0.45);
});

// --- shape handling ---

test("tool schema changes break affinity, because they break the upstream cache anyway", () => {
  const withTools = stablePrefix({ model: "m", tools: TOOLS, messages: [SYSTEM] });
  const without = stablePrefix({ model: "m", messages: [SYSTEM] });
  expect(withTools).not.toBe(without);
});

test("completions endpoint and malformed messages degrade without throwing", () => {
  expect(stablePrefix({ model: "m", prompt: "x".repeat(9000) }).length).toBeLessThan(5000);
  expect(() => stablePrefix({ messages: [null, 42, { noRole: true }] })).not.toThrow();
  expect(stablePrefix({})).toBe("");
});
