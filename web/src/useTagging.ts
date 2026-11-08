import { useCallback, useEffect, useReducer, useRef } from "react";

import { initialQueue, queueReducer, type Item } from "./queue";
import { FORMATS, MAX_BYTES, TaggerError, type Tagger } from "./tagger";
import { formatBytes } from "./tags";

const MAX_IN_FLIGHT = 2; // the engine decodes the next image while the GPU works on the current one

let nextId = 0;

function validate(file: File): string | null {
  if (file.type && !FORMATS.includes(file.type)) return "Unsupported file type";
  if (file.size > MAX_BYTES) return `Larger than ${formatBytes(MAX_BYTES)}`;
  if (file.size === 0) return "Empty file";
  return null;
}

/** The image queue; items wait while `tagger` is null (the model is still loading). */
export function useTagging(tagger: Tagger | null) {
  const [state, dispatch] = useReducer(queueReducer, initialQueue);
  const controllers = useRef(new Map<string, AbortController>());
  const urls = useRef(new Map<string, string>());

  useEffect(() => {
    if (!tagger) return;
    let room = MAX_IN_FLIGHT - state.items.filter((item) => item.status === "running").length;
    for (const item of state.items) {
      if (room <= 0) break;
      if (item.status !== "queued" || controllers.current.has(item.id)) continue;
      room -= 1;
      const controller = new AbortController();
      controllers.current.set(item.id, controller);
      dispatch({ type: "start", id: item.id });
      tagger
        .tag(item.file, controller.signal)
        .then((result) => dispatch({ type: "succeed", id: item.id, result }))
        .catch((err: unknown) => {
          if (controller.signal.aborted) return;
          const message = err instanceof Error ? err.message : "Failed";
          dispatch({ type: "fail", id: item.id, error: message, retryable: err instanceof TaggerError && err.retryable });
        })
        .finally(() => {
          if (controllers.current.get(item.id) === controller) controllers.current.delete(item.id);
        });
    }
  }, [state.items, tagger]);

  useEffect(() => {
    const ctrl = controllers.current;
    const objectUrls = urls.current;
    return () => {
      ctrl.forEach((c) => c.abort());
      objectUrls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);

  const add = useCallback(
    (files: Iterable<File>) => {
      const items: Item[] = [];
      for (const file of files) {
        if (file.type && !file.type.startsWith("image/")) continue;
        const id = `img-${++nextId}`;
        const url = URL.createObjectURL(file);
        urls.current.set(id, url);
        const error = validate(file);
        items.push({
          id,
          file,
          url,
          name: file.name || "Pasted image",
          status: error ? "error" : "queued",
          error: error ?? undefined,
          retryable: false,
        });
      }
      dispatch({ type: "add", items });
      return items.length;
    },
    [],
  );

  const release = (id: string) => {
    controllers.current.get(id)?.abort();
    controllers.current.delete(id);
    const url = urls.current.get(id);
    if (url) URL.revokeObjectURL(url);
    urls.current.delete(id);
  };

  const remove = useCallback((id: string) => {
    release(id);
    dispatch({ type: "remove", id });
  }, []);

  const clear = useCallback(() => {
    [...urls.current.keys()].forEach(release);
    dispatch({ type: "clear" });
  }, []);

  const retry = useCallback((id: string) => dispatch({ type: "requeue", id }), []);
  const select = useCallback((id: string) => dispatch({ type: "select", id }), []);
  const move = useCallback((delta: 1 | -1) => dispatch({ type: "move", delta }), []);

  return { state, add, remove, clear, retry, select, move };
}
