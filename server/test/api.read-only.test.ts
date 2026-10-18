import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createApp, type AppHandle } from "../src/app";
import { VaultError } from "../src/vault/types";

let handle!: AppHandle;
let dir!: string;

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "brain-ro-"));
  await writeFile(join(dir, "Welcome.md"), "# Welcome\n");
  // Folder perms say rw everywhere — read-only must win anyway.
  await mkdir(join(dir, ".brain"), { recursive: true });
  await writeFile(
    join(dir, ".brain", "folder-meta.json"),
    JSON.stringify({ version: 1, icons: {}, colors: {}, mcp: { "": { read: true, write: true } } }),
  );
  handle = createApp({ vaultDir: dir, gitAutocommit: false, readOnly: true });
});
afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
});

async function call(m: string, p: string, b?: unknown) {
  const init: RequestInit = { method: m, headers: {} };
  if (b !== undefined) {
    (init.headers as Record<string, string>)["content-type"] = "application/json";
    init.body = JSON.stringify(b);
  }
  const r = await handle.app.handle(new Request(`http://localhost${p}`, init));
  const text = await r.text();
  return { status: r.status, json: text ? JSON.parse(text) : null };
}

let rpcId = 0;
async function mcp(h: AppHandle, method: string, params: unknown = {}) {
  const r = await h.app.handle(
    new Request("http://localhost/mcp", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
      },
      body: JSON.stringify({ jsonrpc: "2.0", id: ++rpcId, method, params }),
    }),
  );
  return r.json();
}

const INIT_PARAMS = {
  protocolVersion: "2025-03-26",
  capabilities: {},
  clientInfo: { name: "test", version: "0" },
};

async function exists(p: string): Promise<boolean> {
  return stat(p).then(
    () => true,
    () => false,
  );
}

describe("--read-only HTTP API", () => {
  test("server-info reports readOnly", async () => {
    const r = await call("GET", "/api/server-info");
    expect(r.status).toBe(200);
    expect(r.json.readOnly).toBe(true);
    expect(typeof r.json.version).toBe("string");
  });

  test("reads still work", async () => {
    expect((await call("GET", "/api/tree")).status).toBe(200);
    const n = await call("GET", "/api/note/Welcome.md");
    expect(n.status).toBe(200);
    expect(n.json.content).toBe("# Welcome\n");
  });

  test.each([
    ["PUT", "/api/note/Welcome.md", { content: "x" }],
    ["PUT", "/api/note/New.md", { content: "x" }],
    ["DELETE", "/api/note/Welcome.md", undefined],
    ["POST", "/api/folder/sub", undefined],
    ["POST", "/api/rename", { from: "Welcome.md", to: "W.md" }],
    ["PATCH", "/api/settings", { dailyDir: "x" }],
    ["POST", "/api/folder-meta", { path: "a", icon: "star" }],
    ["POST", "/api/folder-mcp-perms", { path: "a", read: true, write: true }],
    ["POST", "/api/git/commit", {}],
    ["POST", "/api/auth/set", { newPassword: "abcd" }],
    ["POST", "/api/auth/keys", { name: "k" }],
  ])("%s %s → 403 READ_ONLY", async (m, p, b) => {
    const r = await call(m, p, b);
    expect(r.status).toBe(403);
    expect(r.json.code).toBe("READ_ONLY");
  });

  test("vault content untouched after blocked writes", async () => {
    await call("PUT", "/api/note/Welcome.md", { content: "x" });
    await call("DELETE", "/api/note/Welcome.md");
    expect(await readFile(join(dir, "Welcome.md"), "utf8")).toBe("# Welcome\n");
    expect(await exists(join(dir, ".brain", "auth.json"))).toBe(false);
  });

  test("login is not blocked by the guard", async () => {
    const r = await call("POST", "/api/auth/login", { password: "nope" });
    expect(r.status).not.toBe(403);
  });

  test("Vault itself refuses writes", async () => {
    await expect(handle.vault.writeNote("x.md", "x")).rejects.toBeInstanceOf(VaultError);
    await expect(handle.vault.writeNote("x.md", "x")).rejects.toMatchObject({
      code: "READ_ONLY",
    });
    await expect(handle.vault.mkdirFolder("f")).rejects.toMatchObject({ code: "READ_ONLY" });
    expect(await exists(join(dir, "x.md"))).toBe(false);
  });
});

describe("--read-only MCP", () => {
  test("write tools are not listed; read tools are", async () => {
    await mcp(handle, "initialize", INIT_PARAMS);
    const res = await mcp(handle, "tools/list");
    const names: string[] = res.result.tools.map((t: { name: string }) => t.name);
    expect(names).toContain("read_note");
    expect(names).toContain("search_notes");
    expect(names).not.toContain("write_note");
    expect(names).not.toContain("append_note");
  });

  test("normal mode still lists write tools", async () => {
    const rw = createApp({ vaultDir: dir, gitAutocommit: false });
    const res = await mcp(rw, "tools/list");
    const names: string[] = res.result.tools.map((t: { name: string }) => t.name);
    expect(names).toContain("write_note");
    expect(names).toContain("append_note");
  });

  test("write_note call fails and creates nothing despite rw folder perms", async () => {
    const res = await mcp(handle, "tools/call", {
      name: "write_note",
      arguments: { path: "Evil.md", content: "x" },
    });
    expect(res.error ?? res.result?.isError).toBeTruthy();
    expect(await exists(join(dir, "Evil.md"))).toBe(false);
  });

  test("instructions carry the read-only notice and seed no file", async () => {
    const res = await mcp(handle, "initialize", INIT_PARAMS);
    expect(res.result.instructions).toContain("READ-ONLY MODE");
    expect(await exists(join(dir, ".brain", "mcp-prompt.md"))).toBe(false);
  });
});
