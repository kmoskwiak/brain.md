#!/usr/bin/env sh
# folio installer for macOS and Linux.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/kmoskwiak/folio/main/install.sh | bash
#
# Env knobs:
#   FOLIO_INSTALL   target directory (default: $HOME/.local/bin)
#   FOLIO_VERSION   release tag to install (default: latest)
#   (BRAIN_INSTALL / BRAIN_VERSION are still honoured as fallbacks.)

set -eu

REPO="kmoskwiak/folio"
BIN_NAME="folio"
INSTALL_DIR="${FOLIO_INSTALL:-${BRAIN_INSTALL:-$HOME/.local/bin}}"
VERSION="${FOLIO_VERSION:-${BRAIN_VERSION:-latest}}"

bold()  { printf "\033[1m%s\033[0m\n" "$1"; }
green() { printf "\033[32m%s\033[0m\n" "$1"; }
warn()  { printf "\033[33m%s\033[0m\n" "$1"; }
die()   { printf "\033[31merror:\033[0m %s\n" "$1" >&2; exit 1; }

# ---------- detect platform ----------
case "$(uname -s)" in
  Linux*)  OS=linux  ;;
  Darwin*) OS=darwin ;;
  *) die "unsupported OS: $(uname -s). Use install.ps1 on Windows." ;;
esac

case "$(uname -m)" in
  x86_64|amd64)  ARCH=x64   ;;
  arm64|aarch64) ARCH=arm64 ;;
  *) die "unsupported architecture: $(uname -m)" ;;
esac

ASSET="folio-${OS}-${ARCH}"

if [ "$VERSION" = "latest" ]; then
  URL="https://github.com/${REPO}/releases/latest/download/${ASSET}"
else
  URL="https://github.com/${REPO}/releases/download/${VERSION}/${ASSET}"
fi

# ---------- prerequisites ----------
if ! command -v curl >/dev/null 2>&1; then
  die "curl is required."
fi

# ---------- download ----------
mkdir -p "$INSTALL_DIR"
TARGET="$INSTALL_DIR/$BIN_NAME"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

bold "→ downloading $ASSET ($VERSION)"
printf "  url: %s\n" "$URL"
if ! curl -fSL --progress-bar "$URL" -o "$TMP"; then
  die "download failed. Check that release '$VERSION' exists at https://github.com/${REPO}/releases"
fi

mv "$TMP" "$TARGET"
chmod +x "$TARGET"
trap - EXIT

# ---------- verify ----------
bold "→ verifying"
if ! "$TARGET" --version >/dev/null 2>&1; then
  die "the binary was downloaded to $TARGET but failed to run ($?)."
fi
VERSION_OUT="$("$TARGET" --version 2>/dev/null | head -1)"

# ---------- summary ----------
printf "\n"
green "✔ installed: $VERSION_OUT"
printf "  location: %s\n\n" "$TARGET"

case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    warn "$INSTALL_DIR is not on your PATH."
    printf "  Add this line to your shell config (~/.zshrc, ~/.bashrc, etc):\n\n"
    printf "    export PATH=\"%s:\$PATH\"\n\n" "$INSTALL_DIR"
    ;;
esac

bold "Get started:"
printf "  folio --help                         # see all flags\n"
printf "  folio                                # serve on :3000\n"
printf "  folio --vault-dir ~/my-notes         # custom vault\n"
printf "\n"
printf "Then open: \033[34mhttp://localhost:3000\033[0m\n"
