import { describe, test, expect } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { RagPipeline } from "../src/rag/pipeline";
import { RagStore } from "../src/rag/store";
import { DEFAULT_RAG_CONFIG } from "../src/rag/types";
import type { Embedder } from "../src/rag/provider";

// Pipeline tests with a stub vault + embedder: no model download, no fs vault.

const fakeEmbedder: Embedder = {
  providerId: "local",
  modelId: DEFAULT_RAG_CONFIG.local.model,
  dim: 4,
  ready: async () => {},
  embed: async (texts) => texts.map(() => Float32Array.from([1, 0, 0, 0])),
};

function makePipeline(
  store: RagStore,
  vault: { listAllNotes: () => Promise<string[]>; readNote: (rel: string) => Promise<unknown> },
): RagPipeline {
  const p = new RagPipeline(
    { ...vault, onMutation: () => () => {} } as unknown as ConstructorParameters<typeof RagPipeline>[0],
    store,
    { ...DEFAULT_RAG_CONFIG, enabled: true },
  );
  (p as unknown as { embedder: Embedder }).embedder = fakeEmbedder;
  return p;
}

async function withStore(fn: (store: RagStore) => Promise<void>): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), "brain-pipeline-"));
  const store = new RagStore(dir, 4);
  try {
    await fn(store);
  } finally {
    await store.close();
    await rm(dir, { recursive: true, force: true });
  }
}

const note = (body: string) => ({ content: body, mtime: 1 });

describe("RagPipeline.reindexAll", () => {
  test("prunes rows of notes no longer in the vault", async () => {
    await withStore(async (store) => {
      await store.open();
      const pipeline = makePipeline(store, {
        listAllNotes: async () => ["a.md"],
        readNote: async () => note("# A\n\nbody"),
      });
      await store.upsert([
        {
          id: "gone.md#0",
          path: "gone.md",
          chunkIndex: 0,
          text: "stale",
          headingTrail: [],
          lineStart: 1,
          lineEnd: 1,
          mtime: 0,
          modelId: "m",
          providerId: "local",
          embedding: Float32Array.from([1, 0, 0, 0]),
        },
      ]);
      await pipeline.reindexAll();
      const paths = (await store.search(Float32Array.from([1, 0, 0, 0]), 50)).map((h) => h.path);
      expect(paths).toEqual(["a.md"]);
    });
  });

  test("keeps a note indexed mid-reindex even if no listing contains it", async () => {
    await withStore(async (store) => {
      await store.open();
      let pipeline!: RagPipeline;
      pipeline = makePipeline(store, {
        // new.md never shows up in a listing (created after the last one).
        listAllNotes: async () => ["a.md"],
        readNote: async (rel: string) => {
          // While a.md is being indexed, a write event indexes new.md.
          if (rel === "a.md") await pipeline.indexNote("new.md");
          return note(`# ${rel}\n\nbody`);
        },
      });
      await pipeline.reindexAll();
      const paths = new Set(
        (await store.search(Float32Array.from([1, 0, 0, 0]), 50)).map((h) => h.path),
      );
      expect(paths).toEqual(new Set(["a.md", "new.md"]));
    });
  });
});

describe("RagPipeline.reindexIfEmpty", () => {
  test("indexes an empty store, skips a non-empty one", async () => {
    await withStore(async (store) => {
      await store.open();
      let listed = 0;
      const pipeline = makePipeline(store, {
        listAllNotes: async () => {
          listed++;
          return ["a.md"];
        },
        readNote: async () => note("# A\n\nbody"),
      });
      pipeline.reindexIfEmpty();
      await Bun.sleep(50);
      expect(await store.countAll()).toBe(1);
      const before = listed;
      pipeline.reindexIfEmpty();
      await Bun.sleep(50);
      expect(listed).toBe(before);
    });
  });
});
