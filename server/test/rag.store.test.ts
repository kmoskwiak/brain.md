import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { RagStore } from "../src/rag/store";
import type { EmbeddedChunk } from "../src/rag/types";

const DIM = 4;

function makeRow(
  path: string,
  idx: number,
  vec: number[],
  modelId = "test-model",
): EmbeddedChunk {
  return {
    id: `${path}#${idx}`,
    path,
    chunkIndex: idx,
    text: `chunk ${idx} of ${path}`,
    headingTrail: ["Heading", `H${idx}`],
    lineStart: idx * 10 + 1,
    lineEnd: idx * 10 + 5,
    mtime: 1700000000 + idx,
    modelId,
    providerId: "local",
    embedding: Float32Array.from(vec),
  };
}

describe("RagStore — V47", () => {
  let dir!: string;
  let store!: RagStore;

  beforeAll(async () => {
    dir = await mkdtemp(join(tmpdir(), "brain-lance-"));
    store = new RagStore(dir, DIM);
    await store.open();
  });

  afterAll(async () => {
    await store.close();
    await rm(dir, { recursive: true, force: true });
  });

  test("empty store countAll == 0", async () => {
    expect(await store.countAll()).toBe(0);
  });

  test("upsert([]) is a no-op", async () => {
    await store.upsert([]);
    expect(await store.countAll()).toBe(0);
  });

  test("upsert rows then countAll matches", async () => {
    await store.upsert([
      makeRow("a.md", 0, [1, 0, 0, 0]),
      makeRow("a.md", 1, [0, 1, 0, 0]),
      makeRow("b.md", 0, [0, 0, 1, 0]),
    ]);
    expect(await store.countAll()).toBe(3);
  });

  test("upsert with same id updates (merge)", async () => {
    await store.upsert([makeRow("a.md", 0, [1, 0, 0, 0], "updated-model")]);
    expect(await store.countAll()).toBe(3);
    const models = await store.distinctProviderModel();
    const ids = models.map((m) => m.modelId).sort();
    expect(ids).toContain("updated-model");
    expect(ids).toContain("test-model");
  });

  test("search returns nearest neighbor first", async () => {
    const hits = await store.search(Float32Array.from([1, 0, 0, 0]), 3);
    expect(hits.length).toBe(3);
    // a.md#0 has embedding [1,0,0,0] → distance 0 → score 1
    expect(hits[0]!.path).toBe("a.md");
    expect(hits[0]!.chunkIndex).toBe(0);
    expect(hits[0]!.score).toBeGreaterThan(0.99);
    expect(hits[0]!.headingTrail).toEqual(["Heading", "H0"]);
  });

  test("deleteByPath removes all rows for that path", async () => {
    await store.deleteByPath("a.md");
    expect(await store.countAll()).toBe(1);
    const hits = await store.search(Float32Array.from([0, 0, 1, 0]), 3);
    expect(hits[0]!.path).toBe("b.md");
  });

  test("deleteByPath SQL-escapes single quotes", async () => {
    await store.upsert([makeRow("o'malley.md", 0, [0, 0, 0, 1])]);
    expect(await store.countAll()).toBe(2);
    await store.deleteByPath("o'malley.md");
    expect(await store.countAll()).toBe(1);
  });

  test("reopen on same dir sees existing rows", async () => {
    await store.close();
    const reopen = new RagStore(dir, DIM);
    await reopen.open();
    expect(await reopen.countAll()).toBe(1);
    await reopen.close();
    // restore the outer store for any later tests
    store = new RagStore(dir, DIM);
    await store.open();
  });

  // ---------------- tasks (V55) ----------------

  function makeTask(
    path: string,
    line: number,
    vec: number[],
    done = false,
  ) {
    return {
      id: `${path}#L${line}`,
      path,
      lineNo: line,
      text: `task at L${line} ${done ? "done" : "open"}`,
      done,
      embedding: Float32Array.from(vec),
      mtime: 1700000000 + line,
      modelId: "test-model",
      providerId: "local" as const,
    };
  }

  test("countTasks starts at 0", async () => {
    expect(await store.countTasks()).toBe(0);
  });

  test("upsertTasks + searchTasks returns nearest first", async () => {
    await store.upsertTasks([
      makeTask("a.md", 1, [1, 0, 0, 0], false),
      makeTask("a.md", 3, [0, 1, 0, 0], true),
      makeTask("b.md", 5, [0, 0, 1, 0], false),
    ]);
    expect(await store.countTasks()).toBe(3);
    const all = await store.searchTasks(Float32Array.from([1, 0, 0, 0]), 3, "all");
    expect(all[0]!.path).toBe("a.md");
    expect(all[0]!.lineNo).toBe(1);
    expect(all[0]!.score).toBeGreaterThan(0.99);
  });

  test("searchTasks filter=open excludes done", async () => {
    // a.md#L3 is done; should not appear
    const open = await store.searchTasks(Float32Array.from([0, 1, 0, 0]), 5, "open");
    expect(open.every((h) => h.done === false)).toBe(true);
    expect(open.find((h) => h.path === "a.md" && h.lineNo === 3)).toBeUndefined();
  });

  test("searchTasks filter=done returns only done rows", async () => {
    const done = await store.searchTasks(Float32Array.from([0, 1, 0, 0]), 5, "done");
    expect(done.every((h) => h.done === true)).toBe(true);
    expect(done[0]!.path).toBe("a.md");
    expect(done[0]!.lineNo).toBe(3);
  });

  test("deleteTasksByPath removes only that note's tasks", async () => {
    await store.deleteTasksByPath("a.md");
    expect(await store.countTasks()).toBe(1);
    const remaining = await store.searchTasks(
      Float32Array.from([0, 0, 1, 0]),
      5,
      "all",
    );
    expect(remaining[0]!.path).toBe("b.md");
  });

  // ---------------- scoring + maintenance ----------------

  test("score is raw cosine: identical ≈ 1, orthogonal ≈ 0, opposite ≈ -1", async () => {
    await store.upsert([
      makeRow("same.md", 0, [0, 1, 0, 0]),
      makeRow("orth.md", 0, [1, 0, 0, 0]),
      makeRow("opp.md", 0, [0, -1, 0, 0]),
    ]);
    const hits = await store.search(Float32Array.from([0, 1, 0, 0]), 10);
    const score = (p: string) => hits.find((h) => h.path === p)!.score;
    expect(score("same.md")).toBeCloseTo(1, 5);
    expect(score("orth.md")).toBeCloseTo(0, 5);
    expect(score("opp.md")).toBeCloseTo(-1, 5);
  });

  test("rows of a different dim are skipped, not scored NaN", async () => {
    await store.upsert([makeRow("old-model.md", 0, [1, 0, 0])]);
    const hits = await store.search(Float32Array.from([1, 0, 0, 0]), 50);
    expect(hits.find((h) => h.path === "old-model.md")).toBeUndefined();
    expect(hits.every((h) => Number.isFinite(h.score))).toBe(true);
    expect((await store.distinctDims()).sort()).toEqual([3, 4]);
  });

  test("pruneExcept removes rows for paths no longer in the vault", async () => {
    const removed = await store.pruneExcept(new Set(["b.md", "same.md"]));
    expect(removed).toBeGreaterThan(0);
    const hits = await store.search(Float32Array.from([0, 1, 0, 0]), 50);
    expect(new Set(hits.map((h) => h.path))).toEqual(new Set(["b.md", "same.md"]));
    const tasks = await store.searchTasks(Float32Array.from([0, 0, 1, 0]), 50, "all");
    expect(tasks.every((t) => t.path === "b.md")).toBe(true);
  });

  test("rows of another dim survive reopen at a new dim (switching back keeps them)", async () => {
    await store.close();
    // File currently holds DIM-length rows. Reopen as an 8-dim store.
    const reopened = new RagStore(dir, DIM);
    reopened.setDim(8);
    await reopened.open();
    const kept = await reopened.countAll();
    expect(kept).toBeGreaterThan(0);
    // ...but they can't be scored against an 8-dim query.
    expect(await reopened.search(new Float32Array(8).fill(0.35), 50)).toEqual([]);
    await reopened.close();
    // Back at DIM: same rows load and score again.
    const back = new RagStore(dir, DIM);
    await back.open();
    expect(await back.countAll()).toBe(kept);
    expect((await back.search(Float32Array.from([0, 1, 0, 0]), 1)).length).toBe(1);
    await back.close();
  });

  test("search excludePath skips that note's chunks before the top-k cut", async () => {
    // Previous test closed the shared store.
    store = new RagStore(dir, DIM);
    await store.open();
    await store.upsert([
      makeRow("seed.md", 0, [0, 1, 0, 0]),
      makeRow("seed.md", 1, [0, 1, 0, 0]),
    ]);
    const hits = await store.search(Float32Array.from([0, 1, 0, 0]), 2, {
      excludePath: "seed.md",
    });
    expect(hits.length).toBe(2);
    expect(hits.some((h) => h.path === "seed.md")).toBe(false);
  });
});

describe("RagStore — format version", () => {
  test("a v1 vectors.json (512-token chunks) is wiped on open", async () => {
    const dir = await mkdtemp(join(tmpdir(), "brain-store-v1-"));
    try {
      await writeFile(
        join(dir, "vectors.json"),
        JSON.stringify({
          version: 1,
          dim: DIM,
          notes: [{ ...makeRow("old.md", 0, [1, 0, 0, 0]), embedding: [1, 0, 0, 0] }],
          tasks: [],
        }),
      );
      const s = new RagStore(dir, DIM);
      await s.open();
      expect(await s.countAll()).toBe(0);
      await s.close();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("minScore / pathFilter apply before top-k; minScore 0 = no cutoff", async () => {
    const dir = await mkdtemp(join(tmpdir(), "brain-store-opts-"));
    const s = new RagStore(dir, DIM);
    try {
      await s.open();
      await s.upsert([
        makeRow("hi.md", 0, [0, 0, 0, 1]),
        makeRow("mid.md", 0, [0, 0, 0.6, 0.8]),
        makeRow("neg.md", 0, [0, 0, 0, -1]),
      ]);
      const q = Float32Array.from([0, 0, 0, 1]);
      const all = await s.search(q, 50, { minScore: 0 });
      expect(all.some((h) => h.path === "neg.md")).toBe(true);
      const floored = await s.search(q, 50, { minScore: 0.5 });
      expect(floored.map((h) => h.path)).toEqual(["hi.md", "mid.md"]);
      const filtered = await s.search(q, 1, { pathFilter: (p) => p !== "hi.md" });
      expect(filtered.map((h) => h.path)).toEqual(["mid.md"]);
    } finally {
      await s.close();
      await rm(dir, { recursive: true, force: true });
    }
  });
});
