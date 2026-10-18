import { Elysia } from "elysia";
import { VERSION } from "../version";

// --view-only: reject every mutating /api/* request up front. Vault-level
// guards (Vault.assertWritable) back this up for user content; this guard
// also covers settings, folder meta/perms, git, RAG admin and auth admin,
// which write to .brain/ directly.
//
// /oauth/* and /mcp live outside /api/ and are untouched — MCP drops its
// write tools on its own (mcp/server.ts).

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

// POSTs that don't change content: session login/logout and the RAG
// context query (a read).
const ALLOWED_POSTS = new Set(["/api/auth/login", "/api/auth/logout", "/api/context"]);

export function viewOnlyGuard() {
  return new Elysia().onRequest(({ request }) => {
    if (SAFE_METHODS.has(request.method)) return;
    const path = new URL(request.url).pathname;
    if (!path.startsWith("/api/")) return;
    if (request.method === "POST" && ALLOWED_POSTS.has(path)) return;
    return new Response(
      JSON.stringify({ error: "server is view-only", code: "VIEW_ONLY" }),
      { status: 403, headers: { "content-type": "application/json" } },
    );
  });
}

// Open (pre-login) so the SPA can pick its layout before auth resolves.
export function serverInfoRoutes(viewOnly: boolean) {
  return new Elysia().get("/api/server-info", () => ({ viewOnly, version: VERSION }));
}
