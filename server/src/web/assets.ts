// ------------------------------------------------------------------
// Committed stub. The release workflow overwrites this file with
// scripts/gen-web-assets.ts right before `bun build --compile`.
//
// Keep it empty in git: a generated copy hard-imports every file in
// web/dist, so `bun run dev` crashes as soon as a rebuild changes the
// hashed filenames. With ASSETS empty, serve.ts falls back to web/dist
// on disk (or the GitHub cache).
// ------------------------------------------------------------------

export const ASSETS: Record<string, string> = {};

export const ASSETS_GENERATED_AT: number | null = null;
