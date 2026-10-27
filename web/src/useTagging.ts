import { useCallback, useEffect, useReducer, useRef } from "react";

import { ApiError, tagImage, type ServerInfo } from "./api";
import { initialQueue, queueReducer, type Item } from "./queue";
import { formatBytes } from "./tags";

const MAX_IN_FLIGHT = 2; // overlaps the next upload with inference
const MAX_BUSY_RETRIES = 5;

let nextId = 0;

function validate(file: File, info: ServerInfo | null): string | null {
  if (!info) return null; // the server validates too
  if (file.type && !info.formats.includes(file.type)) return "Unsupported file type";
  if (file.size > info.max_upload_bytes) return `Larger than ${formatBytes(info.max_upload_bytes)}`;
  if (file.size === 0) return "Empty file";
  return null;
}

export function useTagging(info: ServerInfo | null) {
  const [state, dispatch] = useReducer(queueReducer, initialQueue);
  const controllers = useRef(new Map<string, AbortController>());
  const urls = useRef(new Map<string, string>());

  useEffect(() => {
    let room = MAX_IN_FLIGHT - state.items.filter((item) => item.status === "uploading").length;
    for (const item of state.items) {
      if (room <= 0) break;
      if (item.status !== "queued" || controllers.current.has(item.id)) continue;
      room -= 1;
      const controller = new AbortController();
      controllers.current.set(item.id, controller);
      dispatch({ type: "start", id: item.id });
      tagImage(item.file, controller.signal)
        .then((result) => dispatch({ type: "succeed", id: item.id, result }))
        .catch((err: unknown) => {
          if (controller.signal.aborted) return;
          if (err instanceof ApiError && err.code === "busy" && item.busyRetries < MAX_BUSY_RETRIES) {
            const delay = (err.retryAfter ?? 2) * 1000 * (item.busyRetries + 1);
            setTimeout(() => dispatch({ type: "requeue", id: item.id, busy: true }), delay);
            return;
          }
          const message = err instanceof Error ? err.message : "Failed";
          dispatch({ type: "fail", id: item.id, error: message, retryable: err instanceof ApiError && err.retryable });
        })
        .finally(() => {
          if (controllers.current.get(item.id) === controller) controllers.current.delete(item.id);
        });
    }
  }, [state.items]);

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
        const error = validate(file, info);
        items.push({
          id,
          file,
          url,
          name: file.name || "Pasted image",
          status: error ? "error" : "queued",
          error: error ?? undefined,
          retryable: false,
          busyRetries: 0,
        });
      }
      dispatch({ type: "add", items });
      return items.length;
    },
    [info],
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

  const retry = useCallback((id: string) => dispatch({ type: "requeue", id, busy: false }), []);
  const select = useCallback((id: string) => dispatch({ type: "select", id }), []);
  const move = useCallback((delta: 1 | -1) => dispatch({ type: "move", delta }), []);

  return { state, add, remove, clear, retry, select, move };
}
