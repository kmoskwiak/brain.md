import { createApp } from "./app";
import { loadConfig } from "./config";
import { parseArgs, getUsage, CliError } from "./cli";
import { mkdir } from "node:fs/promises";
import { VERSION } from "./version";
import { describeProvider } from "./rag/provider";

async function main() {
  let action;
  try {
    action = parseArgs(process.argv.slice(2));
  } catch (e) {
    const msg = e instanceof CliError ? e.message : String(e);
    process.stderr.write(`folio: ${msg}\n\nTry 'folio --help' for usage.\n`);
    process.exit(2);
  }

  if (action.kind === "help") {
    process.stdout.write(getUsage());
    process.exit(0);
  }
  if (action.kind === "version") {
    process.stdout.write(`folio ${VERSION}\n`);
    process.exit(0);
  }

  // V44: resolve config with cli > env > XDG default precedence
  const config = loadConfig(action.options);

  // mkdir -p so first run on a fresh XDG default doesn't crash
  await mkdir(config.vaultDir, { recursive: true });

  const {
    app,
    vault,
    index,
    repo,
    autocommit,
    settings,
    ragStore,
    ragPipeline,
    authStore,
    tokenStore,
    apiKeys,
  } = createApp({
    vaultDir: config.vaultDir,
    gitAutocommit: config.gitAutocommit,
    gitDebounceMs: config.gitDebounceMs,
    mcpDisabled: action.options.mcpDisabled,
    viewOnly: action.options.viewOnly,
  });
  const viewOnly = action.options.viewOnly ?? false;

  await authStore.load();
  // B19 + V66: persist session tokens + named API keys to vault.
  await tokenStore.bindVault(vault);
  await apiKeys.bindVault(vault);
  const loaded = await settings.load();
  // settings.json overrides env defaults
  // --view-only: nothing to commit, and `git init` would write into the vault.
  autocommit.setEnabled(viewOnly ? false : loaded.git.autocommit);
  autocommit.setDebounceMs(loaded.git.debounceMs);

  if (autocommit.opts.enabled) {
    try {
      await repo.ensure();
    } catch (e) {
      console.warn("git init failed; autocommit disabled:", e);
      autocommit.setEnabled(false);
    }
  }
  await index.loadOrBuild();

  // T104: RAG startup. open store if enabled; non-blocking initial reindex
  // if empty. Errors surface via console but never crash the server.
  if (loaded.rag.enabled) {
    try {
      ragStore.setDim(describeProvider(loaded.rag).dim);
      await ragStore.open();
      ragPipeline.applyConfig(loaded.rag);
      ragPipeline.start();
      // fire-and-forget initial reindex if empty (V50)
      ragPipeline.reindexIfEmpty();
    } catch (e) {
      console.warn("[rag] startup failed; RAG disabled:", e);
    }
  }

  // Mount the embedded / on-disk / downloaded web client last so it
  // acts as the SPA catch-all for everything not handled by /api or /mcp.
  const { mountEmbeddedWeb } = await import("./web/serve");
  await mountEmbeddedWeb(app);

  app.listen(config.port);
  console.log(
    `folio server :${config.port} → vault ${config.vaultDir}${viewOnly ? " (view-only)" : ""}`,
  );
}

void main();
