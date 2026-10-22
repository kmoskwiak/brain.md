import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createApp, type AppHandle } from "../src/app";

let vaultDir!: string;
let handle!: AppHandle;

beforeAll(async () => {
  vaultDir = await mkdtemp(join(tmpdir(), "brain-settings-"));
  handle = createApp({ vaultDir, gitAutocommit: false });
});

afterAll(async () => {
  await rm(vaultDir, { recursive: true, force: true });
});

async function fetchJson(method: string, path: string, body?: unknown) {
  const init: RequestInit = { method, headers: {} };
  if (body !== undefined) {
    (init.headers as Record<string, string>)["content-type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await handle.app.handle(new Request(`http://localhost${path}`, init));
  const text = await res.text();
  const json = text ? JSON.parse(text) : null;
  return { status: res.status, json };
}

describe("settings PATCH — rag must round-trip (regression)", () => {
  test("toggling rag.enabled persists across a follow-up GET", async () => {
    const before = await fetchJson("GET", "/api/settings");
    expect(before.json.rag.enabled).toBe(false);

    const patched = await fetchJson("PATCH", "/api/settings", {
      rag: { enabled: true },
    });
    expect(patched.status).toBe(200);
    expect(patched.json.rag.enabled).toBe(true);

    const after = await fetchJson("GET", "/api/settings");
    expect(after.json.rag.enabled).toBe(true);
  });

  test("rag.provider + nested local config persists", async () => {
    const patched = await fetchJson("PATCH", "/api/settings", {
      rag: {
        provider: "openai-compat",
        openaiCompat: {
          baseURL: "http://example.test/v1",
          model: "fake-embed",
          dim: 256,
        },
      },
    });
    expect(patched.status).toBe(200);
    expect(patched.json.rag.provider).toBe("openai-compat");
    expect(patched.json.rag.openaiCompat.baseURL).toBe("http://example.test/v1");
    expect(patched.json.rag.openaiCompat.dim).toBe(256);
  });

  test("rag minScore defaults per provider and round-trips; out-of-range is ignored", async () => {
    const before = await fetchJson("GET", "/api/settings");
    expect(before.json.rag.local.minScore).toBe(0.6);

    const patched = await fetchJson("PATCH", "/api/settings", {
      rag: {
        local: { model: "Xenova/bge-small-en-v1.5", dim: 384, minScore: 0.55 },
        openaiCompat: {
          baseURL: "http://example.test/v1",
          model: "fake-embed",
          dim: 256,
          minScore: 7,
        },
      },
    });
    expect(patched.status).toBe(200);
    expect(patched.json.rag.local.minScore).toBe(0.55);
    expect(patched.json.rag.openaiCompat.minScore).toBe(0);

    const after = await fetchJson("GET", "/api/settings");
    expect(after.json.rag.local.minScore).toBe(0.55);
  });

  test("local model/dim are pinned to the bundled bge-small", async () => {
    const patched = await fetchJson("PATCH", "/api/settings", {
      rag: { local: { model: "Xenova/all-MiniLM-L6-v2", dim: 768 } },
    });
    expect(patched.status).toBe(200);
    expect(patched.json.rag.local.model).toBe("Xenova/bge-small-en-v1.5");
    expect(patched.json.rag.local.dim).toBe(384);
  });

  test("minScore can be PATCHed alone; negative is rejected (0 = no cutoff)", async () => {
    const ok = await fetchJson("PATCH", "/api/settings", {
      rag: { openaiCompat: { minScore: 0.3 } },
    });
    expect(ok.status).toBe(200);
    expect(ok.json.rag.openaiCompat.minScore).toBe(0.3);
    expect(ok.json.rag.openaiCompat.baseURL).toBe("http://example.test/v1");

    const neg = await fetchJson("PATCH", "/api/settings", {
      rag: { openaiCompat: { minScore: -0.2 } },
    });
    expect(neg.json.rag.openaiCompat.minScore).toBe(0.3);
  });
});
