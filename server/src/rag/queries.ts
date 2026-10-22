import { encode } from "gpt-tokenizer";
import type { Vault } from "../vault/vault";
import type { VaultIndex } from "../index/index";
import type { RagPipeline } from "./pipeline";
import { chunkNote } from "./chunker";
import type { SearchHit, TaskHit } from "./types";
import type { TaskDoneFilter } from "./store";
import {
  loadFolderMeta,
  resolveFolderPerms,
  type McpFolderPerms,
} from "../api/folder-perms";

// V54: shared query layer for every RAG-derived MCP tool + HTTP route.
// One place for ranking, dedupe, token budgets, folder-perm filtering.
// ∀ functions are read-only: they never mutate the vault or the store.

export interface RagDeps {
  vault: Vault;
  index: VaultIndex;
  pipeline: RagPipeline;
  ragEnabled: () => boolean;
}

export class RagDisabledError extends Error {
  code = "RAG_DISABLED" as const;
  constructor() {
    super("RAG disabled");
    this.name = "RagDisabledError";
  }
}

function assertEnabled(deps: RagDeps): void {
  if (!deps.ragEnabled()) throw new RagDisabledError();
}

async function permsMap(vault: Vault): Promise<Record<string, McpFolderPerms>> {
  return (await loadFolderMeta(vault)).mcp;
}

// V69: optional folder scope. A scope is one or more folder prefixes; a path is
// in scope if it sits under any of them (prefix match → subfolders included).
// Empty/undefined scope = whole vault (no restriction). Agents pass this to
// confine RAG/search to e.g. ["work", "private/Journal"].
export type Scope = string | string[] | undefined;

export function normalizeScope(scope: Scope): string[] {
  if (!scope) return [];
  const arr = Array.isArray(scope) ? scope : [scope];
  return arr
    .map((s) => s.trim().replace(/^\/+|\/+$/g, ""))
    .filter((s) => s.length > 0);
}

export function inScope(path: string, prefixes: string[]): boolean {
  if (prefixes.length === 0) return true;
  return prefixes.some((p) => path === p || path.startsWith(`${p}/`));
}

// V54: only paths read-allowed by MCP folder perms.
// V69: and inside the requested folder scope (if any).
// Passed to the store as `pathFilter` so it applies before the top-k cut.
async function readableFilter(
  vault: Vault,
  prefixes: string[],
): Promise<(path: string) => boolean> {
  const map = await permsMap(vault);
  return (p) => resolveFolderPerms(p, map).read && inScope(p, prefixes);
}

// Keep the best-scoring hit per path, preserving score order.
function bestPerPath<T extends { path: string; score: number }>(rows: T[]): T[] {
  const best = new Map<string, T>();
  for (const h of rows) {
    const prev = best.get(h.path);
    if (!prev || h.score > prev.score) best.set(h.path, h);
  }
  return [...best.values()].sort((a, b) => b.score - a.score);
}

// ---------------- similar ----------------

export async function similarNotes(
  deps: RagDeps,
  q: string,
  k = 5,
  scope?: Scope,
): Promise<SearchHit[]> {
  assertEnabled(deps);
  if (q.trim() === "") return [];
  const [vec] = await deps.pipeline.embed([q]);
  return deps.pipeline.store.search(vec!, k, {
    pathFilter: await readableFilter(deps.vault, normalizeScope(scope)),
    minScore: deps.pipeline.minScore,
  });
}

// ---------------- related ----------------

export interface RelatedHit extends SearchHit {}

export async function related(
  deps: RagDeps,
  path: string,
  k = 5,
  scope?: Scope,
): Promise<RelatedHit[]> {
  assertEnabled(deps);
  let body = "";
  try {
    body = (await deps.vault.readNote(path)).content;
  } catch {
    return [];
  }
  const chunks = chunkNote(path, body);
  if (chunks.length === 0) return [];
  // Use the first chunk as the seed — usually the headline + intro paragraph.
  // Cheaper than averaging across all chunks and good enough for "what is
  // this note about".
  const [vec] = await deps.pipeline.embed([chunks[0]!.text]);
  // All eligible chunks (not top-k): a note can match via many chunks, so
  // only per-path dedupe tells how many distinct notes there are.
  const hits = await deps.pipeline.store.search(vec!, Infinity, {
    excludePath: path,
    pathFilter: await readableFilter(deps.vault, normalizeScope(scope)),
    minScore: deps.pipeline.minScore,
  });
  return bestPerPath(hits).slice(0, k);
}

// ---------------- context ----------------

export interface ContextResult {
  text: string;
  sources: Array<{
    path: string;
    lineStart: number;
    lineEnd: number;
    score: number;
  }>;
  truncated: boolean;
}

export async function contextForQuery(
  deps: RagDeps,
  q: string,
  budgetTokens = 2000,
  scope?: Scope,
): Promise<ContextResult> {
  assertEnabled(deps);
  if (q.trim() === "") return { text: "", sources: [], truncated: false };

  const [vec] = await deps.pipeline.embed([q]);
  // All eligible chunks; the token budget below decides how many are used.
  const candidates = await deps.pipeline.store.search(vec!, Infinity, {
    pathFilter: await readableFilter(deps.vault, normalizeScope(scope)),
    minScore: deps.pipeline.minScore,
  });

  // Greedy pack ordered by score, dedupe by path (highest-score per path).
  const ordered = bestPerPath(candidates);

  const parts: string[] = [];
  const sources: ContextResult["sources"] = [];
  let used = 0;
  let truncated = false;
  for (const h of ordered) {
    const block = `## ${h.path} (L${h.lineStart}–${h.lineEnd})\n\n${h.snippet.trim()}\n`;
    const tokens = encode(block).length;
    if (used + tokens > budgetTokens && parts.length > 0) {
      truncated = true;
      break;
    }
    parts.push(block);
    sources.push({
      path: h.path,
      lineStart: h.lineStart,
      lineEnd: h.lineEnd,
      score: h.score,
    });
    used += tokens;
  }
  return { text: parts.join("\n"), sources, truncated };
}

// ---------------- semantic outline ----------------

export interface OutlineCluster {
  headingTrail: string[];
  representative: string; // best chunk text in the cluster
  chunkCount: number;
  lineStart: number;
  lineEnd: number;
}

export async function semanticOutline(
  deps: RagDeps,
  path: string,
  threshold = 0.7,
): Promise<OutlineCluster[]> {
  assertEnabled(deps);
  // Folder-perm guard: outlining a note we can't read is meaningless.
  const map = await permsMap(deps.vault);
  if (!resolveFolderPerms(path, map).read) return [];

  let body = "";
  try {
    body = (await deps.vault.readNote(path)).content;
  } catch {
    return [];
  }
  const chunks = chunkNote(path, body);
  if (chunks.length === 0) return [];
  if (chunks.length === 1) {
    return [
      {
        headingTrail: chunks[0]!.headingTrail,
        representative: chunks[0]!.text,
        chunkCount: 1,
        lineStart: chunks[0]!.lineStart,
        lineEnd: chunks[0]!.lineEnd,
      },
    ];
  }

  const vecs = await deps.pipeline.embed(chunks.map((c) => c.text));

  // Online clustering: each cluster keeps a centroid; new chunk joins the
  // nearest centroid above threshold or starts a fresh cluster. Simple,
  // deterministic, no k tuning.
  const clusters: Array<{
    centroid: Float32Array;
    members: number[]; // indexes into chunks/vecs
  }> = [];

  for (let i = 0; i < vecs.length; i++) {
    const v = vecs[i]!;
    let bestIdx = -1;
    let bestSim = threshold;
    for (let c = 0; c < clusters.length; c++) {
      const sim = cosine(v, clusters[c]!.centroid);
      if (sim > bestSim) {
        bestSim = sim;
        bestIdx = c;
      }
    }
    if (bestIdx === -1) {
      clusters.push({ centroid: new Float32Array(v), members: [i] });
    } else {
      const cl = clusters[bestIdx]!;
      cl.members.push(i);
      // Update centroid = running mean.
      const n = cl.members.length;
      for (let d = 0; d < cl.centroid.length; d++) {
        cl.centroid[d] = (cl.centroid[d]! * (n - 1) + v[d]!) / n;
      }
    }
  }

  return clusters.map((cl) => {
    // Representative = member with highest cosine to centroid.
    let bestIdx = cl.members[0]!;
    let bestSim = -Infinity;
    for (const m of cl.members) {
      const sim = cosine(vecs[m]!, cl.centroid);
      if (sim > bestSim) {
        bestSim = sim;
        bestIdx = m;
      }
    }
    const ch = chunks[bestIdx]!;
    // Union the line span across members.
    const ls = cl.members.map((m) => chunks[m]!.lineStart);
    const le = cl.members.map((m) => chunks[m]!.lineEnd);
    return {
      headingTrail: ch.headingTrail,
      representative: ch.text,
      chunkCount: cl.members.length,
      lineStart: Math.min(...ls),
      lineEnd: Math.max(...le),
    };
  });
}

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i]!;
    const bi = b[i]!;
    dot += ai * bi;
    na += ai * ai;
    nb += bi * bi;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

// ---------------- orphans ----------------

export interface OrphanNote {
  path: string;
  isolation: number; // 1 = totally isolated, 0 = highly connected
  inboundLinks: number;
}

// Default isolation cutoff = 1 - minScore: a note is an orphan exactly when
// its nearest neighbour wouldn't make the Related list. Without a floor
// (minScore 0) fall back to 0.35, i.e. nearest neighbour ≤ 0.65.
export function defaultMinIsolation(minScore: number): number {
  return minScore > 0 ? 1 - minScore : 0.35;
}

export async function orphans(
  deps: RagDeps,
  limit = 10,
  minIsolation?: number,
  scope?: Scope,
): Promise<OrphanNote[]> {
  assertEnabled(deps);
  const cutoff = minIsolation ?? defaultMinIsolation(deps.pipeline.minScore);
  const map = await permsMap(deps.vault);
  const prefixes = normalizeScope(scope);
  const entries = deps.index.entries();
  const allPaths = entries.map((e) => e.path);

  // Count inbound backlinks for every note via the existing index.
  const inbound = new Map<string, number>();
  for (const p of allPaths) {
    inbound.set(p, deps.index.backlinks(p).length);
  }
  const noBacklinks = allPaths.filter(
    (p) =>
      (inbound.get(p) ?? 0) === 0 &&
      resolveFolderPerms(p, map).read &&
      inScope(p, prefixes),
  );
  if (noBacklinks.length === 0) return [];

  // For each candidate, find its max cosine sim to any OTHER note.
  // Cheap heuristic: embed the first chunk only.
  const seeds = await Promise.all(
    noBacklinks.map(async (p) => {
      const e = entries.find((x) => x.path === p);
      const body = e ? e.body : "";
      const chs = chunkNote(p, body);
      return chs[0]?.text ?? "";
    }),
  );
  const validIdx = seeds.map((s, i) => (s ? i : -1)).filter((i) => i >= 0);
  if (validIdx.length === 0) return [];

  const vecs = await deps.pipeline.embed(validIdx.map((i) => seeds[i]!));
  const out: OrphanNote[] = [];
  for (let n = 0; n < validIdx.length; n++) {
    const p = noBacklinks[validIdx[n]!]!;
    // Nearest neighbour other than itself. No minScore here: isolation needs
    // the true top score, however low.
    const [nearest] = await deps.pipeline.store.search(vecs[n]!, 1, {
      excludePath: p,
    });
    const topSim = Math.max(0, nearest?.score ?? 0);
    const isolation = 1 - topSim;
    if (isolation >= cutoff) {
      out.push({
        path: p,
        isolation,
        inboundLinks: inbound.get(p) ?? 0,
      });
    }
  }
  out.sort((a, b) => b.isolation - a.isolation);
  return out.slice(0, limit);
}

// ---------------- weekly digest ----------------

const DURATION_RE = /^(\d+)\s*(m|h|d|w)$/i;
function parseSince(since: string): number {
  const m = DURATION_RE.exec(since.trim());
  if (!m) return Date.now() - 7 * 24 * 3600 * 1000;
  const n = Number(m[1]);
  const unit = m[2]!.toLowerCase();
  const mult =
    unit === "m" ? 60 * 1000
    : unit === "h" ? 3600 * 1000
    : unit === "d" ? 24 * 3600 * 1000
    : 7 * 24 * 3600 * 1000;
  return Date.now() - n * mult;
}

export interface DigestCluster {
  topicLabel: string;
  paths: string[];
  representative: string;
}

export async function weeklyDigest(
  deps: RagDeps,
  since = "7d",
  threshold = 0.6,
  scope?: Scope,
): Promise<DigestCluster[]> {
  assertEnabled(deps);
  const cutoff = parseSince(since);
  const map = await permsMap(deps.vault);
  const prefixes = normalizeScope(scope);
  const entries = deps.index.entries().filter((e) => {
    if (!resolveFolderPerms(e.path, map).read) return false;
    if (!inScope(e.path, prefixes)) return false;
    return (e.mtime ?? 0) >= cutoff;
  });
  if (entries.length === 0) return [];

  const allChunks: Array<{ path: string; text: string }> = [];
  for (const e of entries) {
    const chs = chunkNote(e.path, e.body);
    for (const c of chs) allChunks.push({ path: e.path, text: c.text });
  }
  if (allChunks.length === 0) return [];

  const vecs = await deps.pipeline.embed(allChunks.map((c) => c.text));

  // Same online-clustering pass as outline; aggregate paths instead of lines.
  const clusters: Array<{
    centroid: Float32Array;
    members: number[];
  }> = [];
  for (let i = 0; i < vecs.length; i++) {
    const v = vecs[i]!;
    let bestIdx = -1;
    let bestSim = threshold;
    for (let c = 0; c < clusters.length; c++) {
      const sim = cosine(v, clusters[c]!.centroid);
      if (sim > bestSim) {
        bestSim = sim;
        bestIdx = c;
      }
    }
    if (bestIdx === -1) {
      clusters.push({ centroid: new Float32Array(v), members: [i] });
    } else {
      const cl = clusters[bestIdx]!;
      cl.members.push(i);
      const n = cl.members.length;
      for (let d = 0; d < cl.centroid.length; d++) {
        cl.centroid[d] = (cl.centroid[d]! * (n - 1) + v[d]!) / n;
      }
    }
  }

  return clusters
    .map((cl) => {
      // Representative = chunk text closest to centroid.
      let bestIdx = cl.members[0]!;
      let bestSim = -Infinity;
      for (const m of cl.members) {
        const sim = cosine(vecs[m]!, cl.centroid);
        if (sim > bestSim) {
          bestSim = sim;
          bestIdx = m;
        }
      }
      const rep = allChunks[bestIdx]!;
      const paths = Array.from(
        new Set(cl.members.map((m) => allChunks[m]!.path)),
      );
      // Cheap label = first ~80 chars of representative.
      const labelSrc = rep.text.split("\n")[0] ?? rep.text;
      return {
        topicLabel: labelSrc.slice(0, 80),
        paths,
        representative: rep.text,
      };
    })
    .sort((a, b) => b.paths.length - a.paths.length);
}

// ---------------- compare ----------------

export interface CompareResult {
  cosine: number;
  unifiedDiff: string;
  sharedHeadings: string[];
}

function unifiedDiffNaive(a: string, b: string): string {
  const A = a.split(/\r?\n/);
  const B = b.split(/\r?\n/);
  const out: string[] = [];
  const max = Math.max(A.length, B.length);
  for (let i = 0; i < max; i++) {
    const la = A[i];
    const lb = B[i];
    if (la === lb) continue;
    if (la !== undefined) out.push(`-${la}`);
    if (lb !== undefined) out.push(`+${lb}`);
  }
  return out.join("\n");
}

export async function compareNotes(
  deps: RagDeps,
  a: string,
  b: string,
): Promise<CompareResult> {
  assertEnabled(deps);
  const map = await permsMap(deps.vault);
  if (!resolveFolderPerms(a, map).read || !resolveFolderPerms(b, map).read) {
    throw new Error(`read denied for ${a} or ${b}`);
  }
  const [na, nb] = await Promise.all([
    deps.vault.readNote(a),
    deps.vault.readNote(b),
  ]);
  const ca = chunkNote(a, na.content);
  const cb = chunkNote(b, nb.content);
  const sharedHeadings: string[] = [];
  if (ca.length > 0 && cb.length > 0) {
    const headsA = new Set(ca.flatMap((c) => c.headingTrail));
    for (const h of cb.flatMap((c) => c.headingTrail)) {
      if (headsA.has(h)) sharedHeadings.push(h);
    }
  }

  let cos = 0;
  if (ca.length > 0 && cb.length > 0) {
    const [va, vb] = await deps.pipeline.embed([ca[0]!.text, cb[0]!.text]);
    cos = cosine(va!, vb!);
  }

  return {
    cosine: Math.max(0, Math.min(1, cos)),
    unifiedDiff: unifiedDiffNaive(na.content, nb.content),
    sharedHeadings: Array.from(new Set(sharedHeadings)),
  };
}

// ---------------- similar tasks ----------------

export async function similarTasks(
  deps: RagDeps,
  query: string,
  k = 10,
  filter: TaskDoneFilter = "open",
  scope?: Scope,
): Promise<TaskHit[]> {
  assertEnabled(deps);
  if (query.trim() === "") return [];
  const [vec] = await deps.pipeline.embed([query]);
  return deps.pipeline.store.searchTasks(vec!, k, filter, {
    pathFilter: await readableFilter(deps.vault, normalizeScope(scope)),
    minScore: deps.pipeline.minScore,
  });
}
