import { useCallback, useState } from "react";

// Edit-mode "Preview only" toggle (editor column hidden). Persisted to
// localStorage (device-local, NOT vault). Value = "1"|"0".
const KEY = "brain.previewOnly";

function readStored(): boolean {
  try {
    return localStorage.getItem(KEY) === "1";
  } catch {
    // ignore
  }
  return false;
}

export function usePreviewOnly(): [boolean, (v: boolean) => void] {
  const [previewOnly, setPreviewOnlyState] = useState<boolean>(() => readStored());

  const setPreviewOnly = useCallback((v: boolean) => {
    try {
      localStorage.setItem(KEY, v ? "1" : "0");
    } catch {
      // ignore
    }
    setPreviewOnlyState(v);
  }, []);

  return [previewOnly, setPreviewOnly];
}
