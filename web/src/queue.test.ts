import { describe, expect, it } from "vitest";

import { initialQueue, queueReducer, type Item } from "./queue";

const item = (id: string): Item => ({
  id,
  file: new File(["x"], `${id}.png`, { type: "image/png" }),
  name: id,
  url: `blob:${id}`,
  status: "queued",
});

const three = queueReducer(initialQueue, { type: "add", items: [item("a"), item("b"), item("c")] });

describe("queueReducer", () => {
  it("selects the first of newly added items", () => {
    const more = queueReducer(three, { type: "add", items: [item("d"), item("e")] });
    expect(more.selectedId).toBe("d");
  });

  it("moves the selection to the next item when the selected one is removed, or the previous at the end", () => {
    const middle = queueReducer({ ...three, selectedId: "b" }, { type: "remove", id: "b" });
    expect(middle.selectedId).toBe("c");
    const last = queueReducer({ ...three, selectedId: "c" }, { type: "remove", id: "c" });
    expect(last.selectedId).toBe("b");
    const other = queueReducer({ ...three, selectedId: "a" }, { type: "remove", id: "c" });
    expect(other.selectedId).toBe("a");
  });

  it("clamps keyboard movement at both ends", () => {
    expect(queueReducer({ ...three, selectedId: "c" }, { type: "move", delta: 1 }).selectedId).toBe("c");
    expect(queueReducer({ ...three, selectedId: "a" }, { type: "move", delta: -1 }).selectedId).toBe("a");
  });
});
