import type { TagResult } from "./api";

export type Status = "queued" | "uploading" | "done" | "error";

export interface Item {
  id: string;
  file: File;
  name: string;
  /** Object URL for previews; revoked when the item is removed. */
  url: string;
  status: Status;
  result?: TagResult;
  error?: string;
  /** Whether a manual retry can help. */
  retryable?: boolean;
  /** Automatic retries spent on 503 answers. */
  busyRetries: number;
}

export interface QueueState {
  items: Item[];
  selectedId: string | null;
}

export type Action =
  | { type: "add"; items: Item[] }
  | { type: "start"; id: string }
  | { type: "succeed"; id: string; result: TagResult }
  | { type: "fail"; id: string; error: string; retryable: boolean }
  | { type: "requeue"; id: string; busy: boolean }
  | { type: "remove"; id: string }
  | { type: "clear" }
  | { type: "select"; id: string }
  | { type: "move"; delta: 1 | -1 };

export const initialQueue: QueueState = { items: [], selectedId: null };

function update(state: QueueState, id: string, patch: (item: Item) => Item): QueueState {
  return { ...state, items: state.items.map((item) => (item.id === id ? patch(item) : item)) };
}

export function queueReducer(state: QueueState, action: Action): QueueState {
  switch (action.type) {
    case "add":
      if (action.items.length === 0) return state;
      return { items: [...state.items, ...action.items], selectedId: action.items[0]?.id ?? state.selectedId };
    case "start":
      return update(state, action.id, (item) => ({ ...item, status: "uploading", error: undefined }));
    case "succeed":
      return update(state, action.id, (item) => ({ ...item, status: "done", result: action.result }));
    case "fail":
      return update(state, action.id, (item) => ({
        ...item,
        status: "error",
        error: action.error,
        retryable: action.retryable,
      }));
    case "requeue":
      return update(state, action.id, (item) => ({
        ...item,
        status: "queued",
        error: undefined,
        busyRetries: action.busy ? item.busyRetries + 1 : 0,
      }));
    case "remove": {
      const index = state.items.findIndex((item) => item.id === action.id);
      if (index < 0) return state;
      const items = state.items.filter((item) => item.id !== action.id);
      const selectedId =
        state.selectedId === action.id ? (items[Math.min(index, items.length - 1)]?.id ?? null) : state.selectedId;
      return { items, selectedId };
    }
    case "clear":
      return initialQueue;
    case "select":
      return state.items.some((item) => item.id === action.id) ? { ...state, selectedId: action.id } : state;
    case "move": {
      const index = state.items.findIndex((item) => item.id === state.selectedId);
      const next = state.items[Math.max(0, Math.min(state.items.length - 1, index + action.delta))];
      return next ? { ...state, selectedId: next.id } : state;
    }
  }
}
