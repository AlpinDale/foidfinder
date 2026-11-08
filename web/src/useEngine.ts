import { useCallback, useEffect, useState } from "react";

import { LocalTagger, webgpuAvailable } from "./engine/client";
import type { Tagger } from "./tagger";

export interface Engine {
  /** Set once the model is loaded. */
  tagger: Tagger | null;
  /** Bytes of the model downloaded so far, while it loads. */
  download: { loaded: number; total: number } | null;
  /** Why the model cannot run in this browser. */
  error: string | null;
  retry: () => void;
}

const NO_WEBGPU = "This browser has no WebGPU, which foidfinder needs to run the model. Recent Chrome, Edge and Safari have it.";

/** Loads the model into this browser's GPU (downloaded once, then cached). */
export function useEngine(): Engine {
  const supported = webgpuAvailable();
  const [tagger, setTagger] = useState<Tagger | null>(null);
  const [download, setDownload] = useState<Engine["download"]>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!supported) return;
    let stopped = false;
    const engine = new LocalTagger((loaded, total) => setDownload({ loaded, total }));
    engine.ready
      .then(() => {
        if (stopped) return;
        setTagger(engine);
        setDownload(null);
      })
      .catch((err: unknown) => {
        if (stopped) return;
        setDownload(null);
        setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      stopped = true;
      engine.terminate();
    };
  }, [attempt, supported]);

  const retry = useCallback(() => {
    setError(null);
    setAttempt((n) => n + 1);
  }, []);
  return { tagger, download, error: supported ? error : NO_WEBGPU, retry };
}
