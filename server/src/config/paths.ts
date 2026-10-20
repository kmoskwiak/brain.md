import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

// V44: XDG Base Directory Spec applied uniformly across macOS, Linux,
// Windows — no platform branching. Users get a single predictable
// default location regardless of OS.

const APP_DIR = "folio";
// Directory name used before the fork was renamed. Kept as a fallback so
// existing installs keep finding their vault and settings.
const LEGACY_APP_DIR = "brain.md";

function xdg(envName: string, fallbackSubpath: string): string {
  const env = process.env[envName];
  if (env && env.trim() !== "") return env;
  return join(homedir(), fallbackSubpath);
}

function appDir(base: string): string {
  const current = join(base, APP_DIR);
  const legacy = join(base, LEGACY_APP_DIR);
  if (!existsSync(current) && existsSync(legacy)) return legacy;
  return current;
}

export function getDefaultVaultDir(): string {
  // $XDG_DATA_HOME defaults to $HOME/.local/share per spec
  return join(appDir(xdg("XDG_DATA_HOME", ".local/share")), "vault");
}

export function getDefaultSettingsDir(): string {
  // $XDG_CONFIG_HOME defaults to $HOME/.config per spec
  return appDir(xdg("XDG_CONFIG_HOME", ".config"));
}
