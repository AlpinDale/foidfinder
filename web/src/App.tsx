import { useEffect, useRef, useState } from "react";

import { QueueList } from "./components/QueueList";
import { Results } from "./components/Results";
import { FORMATS, MAX_BYTES } from "./tagger";
import { formatBytes } from "./tags";
import { useEngine } from "./useEngine";
import { useTagging } from "./useTagging";

const FORMAT_NAMES: Record<string, string> = {
  "image/jpeg": "JPEG",
  "image/png": "PNG",
  "image/webp": "WebP",
  "image/gif": "GIF",
  "image/avif": "AVIF",
  "image/bmp": "BMP",
};

function isTyping(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));
}

export function App() {
  const engine = useEngine();
  const { download } = engine;
  const [notice, setNotice] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const { state, add, remove, clear, retry, select, move } = useTagging(engine.tagger);
  const selected = state.items.find((item) => item.id === state.selectedId) ?? null;

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(null), 5000);
    return () => clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    let depth = 0;
    const hasFiles = (e: DragEvent) => e.dataTransfer?.types.includes("Files") ?? false;
    const onPaste = (e: ClipboardEvent) => {
      if (isTyping(e.target)) return;
      const files = [...(e.clipboardData?.files ?? [])];
      if (files.length) {
        e.preventDefault();
        if (!add(files)) setNotice("No image in clipboard");
      }
    };
    const onDragEnter = (e: DragEvent) => {
      if (!hasFiles(e) && !e.dataTransfer?.types.includes("text/uri-list")) return;
      e.preventDefault();
      depth += 1;
      setDragging(true);
    };
    const onDragLeave = () => {
      depth = Math.max(0, depth - 1);
      if (depth === 0) setDragging(false);
    };
    const onDragOver = (e: DragEvent) => e.preventDefault();
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      depth = 0;
      setDragging(false);
      const files = [...(e.dataTransfer?.files ?? [])];
      if (files.length === 0) setNotice("Save the image first, then drop the file");
      else if (!add(files)) setNotice("No images in drop");
    };
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "ArrowDown" || e.key === "j") move(1);
      else if (e.key === "ArrowUp" || e.key === "k") move(-1);
      else if ((e.key === "Delete" || e.key === "Backspace") && state.selectedId) remove(state.selectedId);
      else return;
      e.preventDefault();
    };
    document.addEventListener("paste", onPaste);
    document.addEventListener("dragenter", onDragEnter);
    document.addEventListener("dragleave", onDragLeave);
    document.addEventListener("dragover", onDragOver);
    document.addEventListener("drop", onDrop);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("paste", onPaste);
      document.removeEventListener("dragenter", onDragEnter);
      document.removeEventListener("dragleave", onDragLeave);
      document.removeEventListener("dragover", onDragOver);
      document.removeEventListener("drop", onDrop);
      document.removeEventListener("keydown", onKey);
    };
  }, [add, move, remove, state.selectedId]);

  const formats = FORMATS.map((f) => FORMAT_NAMES[f]).filter(Boolean);

  return (
    <div className="app">
      <header className="topbar">
        <span className="wordmark">foidfinder</span>
        <button type="button" className="button" onClick={() => input.current?.click()}>
          Add images
        </button>
        <input
          ref={input}
          type="file"
          accept={FORMATS.join(",")}
          multiple
          hidden
          data-testid="file-input"
          onChange={(e) => {
            add(e.target.files ?? []);
            e.target.value = "";
          }}
        />
      </header>
      {engine.error && (
        <div className="banner" role="alert">
          <span>{engine.error}</span>
          <button type="button" className="button small" onClick={engine.retry}>
            Retry
          </button>
        </div>
      )}
      {download && (
        <div className="banner neutral" role="status">
          <span>
            {download.loaded < download.total
              ? `Downloading the model, ${formatBytes(download.loaded)} of ${formatBytes(download.total)}. Later visits load it from this browser.`
              : "Preparing the model"}
          </span>
          <progress max={download.total} value={download.loaded} />
        </div>
      )}
      {notice && (
        <div className="banner neutral" role="status">
          {notice}
        </div>
      )}
      {selected ? (
        <main className="workspace">
          <QueueList
            items={state.items}
            selectedId={state.selectedId}
            onSelect={select}
            onRemove={remove}
            onClear={clear}
          />
          <figure className="viewer">
            <img src={selected.url} alt={selected.name} />
          </figure>
          <div className="results" aria-live="polite">
            <Results item={selected} onRetry={retry} />
          </div>
        </main>
      ) : (
        <main className="empty">
          <button type="button" className="dropzone" onClick={() => input.current?.click()}>
            <span className="dropzone-title">Drop or paste images</span>
            <span className="muted">
              {formats.join(", ")} up to {formatBytes(MAX_BYTES)}
            </span>
          </button>
        </main>
      )}
      {dragging && (
        <div className="drop-overlay" aria-hidden="true">
          Drop to identify
        </div>
      )}
    </div>
  );
}
