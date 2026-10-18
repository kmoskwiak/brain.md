import { describe, test, expect } from "bun:test";
import { parseArgs } from "../src/cli";

describe("CLI --mcp-disabled — T123 / V46", () => {
  test("default = mcpDisabled undefined (MCP mounted)", () => {
    const r = parseArgs([]);
    expect(r.kind).toBe("run");
    if (r.kind === "run") {
      expect(r.options.mcpDisabled).toBeUndefined();
    }
  });

  test("--mcp-disabled sets the flag", () => {
    const r = parseArgs(["--mcp-disabled"]);
    expect(r.kind).toBe("run");
    if (r.kind === "run") {
      expect(r.options.mcpDisabled).toBe(true);
    }
  });

  test("combines with --port", () => {
    const r = parseArgs(["--mcp-disabled", "--port", "4000"]);
    if (r.kind === "run") {
      expect(r.options.mcpDisabled).toBe(true);
      expect(r.options.port).toBe(4000);
    }
  });
});

describe("CLI --read-only", () => {
  test("default = readOnly undefined", () => {
    const r = parseArgs([]);
    if (r.kind === "run") expect(r.options.readOnly).toBeUndefined();
  });

  test("--read-only sets the flag", () => {
    const r = parseArgs(["--read-only"]);
    expect(r.kind).toBe("run");
    if (r.kind === "run") expect(r.options.readOnly).toBe(true);
  });

  test("combines with --port and --vault-dir", () => {
    const r = parseArgs(["--read-only", "--port", "4000", "-v", "/tmp/v"]);
    expect(r).toEqual({
      kind: "run",
      options: { readOnly: true, port: 4000, vaultDir: "/tmp/v" },
    });
  });
});
