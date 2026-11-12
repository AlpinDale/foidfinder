// The result cards the content script shows at the bottom-right of a page, in a closed shadow root so neither the
// page's styles nor its scripts reach them. Text goes in through textContent only.

import type { Tag } from "../tagger";
import { danbooruUrl, formatBytes, formatDuration, percent, splitTag } from "../tags";
import type { CardState } from "./messages";

const MAX_CARDS = 3;

const CSS = `
:host { all: initial; }
.stack { display: flex; flex-direction: column; gap: 8px; align-items: flex-end; }
.card {
  --bg: #ffffff; --text: #121214; --muted: #696970; --line: #e1e1e4; --track: #e4e4e7; --accent: #1f4fd8; --error: #c4291c;
  width: 300px; box-sizing: border-box; padding: 10px 12px 12px; border-radius: 10px;
  background: var(--bg); color: var(--text); border: 1px solid var(--line);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.28);
  font: 13px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; text-align: left;
}
@media (prefers-color-scheme: dark) {
  .card { --bg: #1b1b1e; --text: #ececef; --muted: #9a9aa3; --line: #2b2b30; --track: #2e2e33; --accent: #7a9dff; --error: #ff6b5e; }
}
.head { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.thumb { width: 28px; height: 28px; border-radius: 5px; object-fit: cover; flex: none; background: var(--track); }
.title { font-weight: 600; flex: 1; }
.close {
  all: unset; cursor: pointer; width: 22px; height: 22px; border-radius: 5px; text-align: center;
  line-height: 22px; font-size: 16px; color: var(--muted);
}
.close:hover { background: var(--track); color: var(--text); }
.close:focus-visible { outline: 2px solid var(--accent); }
.row { margin-top: 6px; }
.label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
.match { display: flex; align-items: baseline; gap: 6px; }
.match a { color: var(--text); text-decoration: none; font-weight: 600; font-size: 15px; }
.match a:hover { text-decoration: underline; }
.match.minor a { font-weight: 400; font-size: 13px; color: var(--muted); }
.qualifier { color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.score { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
.none { color: var(--muted); }
.status { color: var(--muted); }
.error { color: var(--error); }
.bar { height: 4px; border-radius: 2px; background: var(--track); margin-top: 8px; overflow: hidden; }
.bar > span { display: block; height: 100%; background: var(--accent); }
.foot { margin-top: 8px; font-size: 11px; color: var(--muted); }
.upgrade { all: unset; cursor: pointer; display: block; margin-top: 8px; font-size: 12px; color: var(--accent); }
.upgrade:hover { text-decoration: underline; }
.upgrade:focus-visible { outline: 2px solid var(--accent); }
`;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className = "", ...children: (Node | string)[]): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  e.append(...children);
  return e;
}

function match(entry: Tag, minor: boolean): HTMLElement {
  const { name, qualifier } = splitTag(entry.tag);
  const link = el("a", "", name);
  link.href = danbooruUrl(entry.tag);
  link.target = "_blank";
  link.rel = "noreferrer";
  link.title = entry.tag;
  return el("div", minor ? "match minor" : "match", link, el("span", "qualifier", qualifier), el("span", "score", percent(entry.score)));
}

/** Accepted tags, or else the best guesses marked as possible, or else "No match" (omitted for optional rows). */
function row(label: string, accepted: Tag[], candidates: Tag[], optional: boolean): HTMLElement | null {
  if (accepted.length === 0 && candidates.length === 0 && optional) return null;
  const r = el("div", "row", el("div", "label", accepted.length ? label : `${label}, possibly`));
  if (accepted.length) r.append(...accepted.slice(0, 3).map((t) => match(t, false)));
  else if (candidates.length) r.append(...candidates.slice(0, 2).map((t) => match(t, true)));
  else r.append(el("div", "none", "No match"));
  return r;
}

function upgradeButton(onUpgrade: () => void): HTMLElement {
  const b = el("button", "upgrade", "Allow full-quality downloads from all sites");
  b.type = "button";
  b.addEventListener("click", onUpgrade);
  return b;
}

/** The body of a card for one state; `onUpgrade` opens the setting that allows downloads from all sites. */
export function cardBody(card: CardState, onUpgrade: () => void = () => {}): HTMLElement[] {
  switch (card.state) {
    case "loading":
      return [el("p", "status", "Identifying…")];
    case "download": {
      if (card.loaded >= card.total) return [el("p", "status", "Loading the model…")];
      const bar = el("div", "bar", el("span"));
      const fill = bar.firstElementChild as HTMLElement;
      fill.style.width = `${(100 * card.loaded) / card.total}%`;
      return [
        el("p", "status", `Downloading the model, ${formatBytes(card.loaded)} of ${formatBytes(card.total)}. This happens once.`),
        bar,
      ];
    }
    case "error":
      return [el("p", "error", `Couldn't identify this image. ${card.message}`), ...(card.upgrade ? [upgradeButton(onUpgrade)] : [])];
    case "result": {
      const { tags, candidates, timings_ms } = card.result;
      const rows = [
        row("Character", tags.character ?? [], candidates.character ?? [], false),
        row("Series", tags.copyright ?? [], candidates.copyright ?? [], false),
        row("Artist", tags.artist ?? [], candidates.artist ?? [], true),
      ].filter((r): r is HTMLElement => r !== null);
      const foot = `On your GPU in ${formatDuration(timings_ms.total)}${card.screenshot ? ", from a screenshot of the page" : ""}`;
      return [...rows, el("div", "foot", foot), ...(card.upgrade ? [upgradeButton(onUpgrade)] : [])];
    }
  }
}

/** The cards of one page: newest at the bottom, at most three, closed with × or Escape. */
export class CardStack {
  private host: HTMLElement | null = null;
  private stack: HTMLElement | null = null;
  private cards = new Map<number, { card: HTMLElement; body: HTMLElement }>();

  constructor(private onUpgrade: () => void) {}

  private mount(): HTMLElement {
    if (this.stack) return this.stack;
    // An unknown element name: page rules for div, span and the like do not match it.
    this.host = document.createElement("foidfinder-cards");
    this.host.setAttribute(
      "style",
      "position: fixed !important; right: 16px !important; bottom: 16px !important; z-index: 2147483647 !important; " +
        "display: block !important; visibility: visible !important; margin: 0 !important; padding: 0 !important;",
    );
    const root = this.host.attachShadow({ mode: "closed" });
    const style = document.createElement("style");
    style.textContent = CSS;
    this.stack = el("div", "stack");
    this.stack.setAttribute("role", "status");
    this.stack.setAttribute("aria-live", "polite");
    root.append(style, this.stack);
    document.documentElement.append(this.host);
    window.addEventListener("keydown", this.onKey, true);
    return this.stack;
  }

  private onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape" && this.cards.size) this.clear();
  };

  show(id: number, image: string, state: CardState) {
    const stack = this.mount();
    this.setHidden(false);
    let entry = this.cards.get(id);
    if (!entry) {
      const thumb = el("img", "thumb");
      thumb.alt = "";
      thumb.addEventListener("error", () => thumb.remove()); // e.g. a blob: URL the page has revoked
      thumb.src = image;
      const close = el("button", "close", "×");
      close.type = "button";
      close.setAttribute("aria-label", "Close");
      close.addEventListener("click", () => this.close(id));
      const body = el("div");
      const card = el("div", "card", el("div", "head", thumb, el("span", "title", "foidfinder"), close), body);
      stack.append(card);
      entry = { card, body };
      this.cards.set(id, entry);
      while (this.cards.size > MAX_CARDS) {
        const oldest = this.cards.keys().next().value;
        if (oldest === undefined) break;
        this.close(oldest);
      }
    }
    entry.body.replaceChildren(...cardBody(state, this.onUpgrade));
  }

  /** Hidden while the page is screenshotted, so the cards do not cover the image. */
  setHidden(hidden: boolean) {
    if (this.host) this.host.style.setProperty("visibility", hidden ? "hidden" : "visible", "important");
  }

  close(id: number) {
    this.cards.get(id)?.card.remove();
    this.cards.delete(id);
    if (this.cards.size === 0) this.clear();
  }

  clear() {
    this.cards.clear();
    this.host?.remove();
    this.host = null;
    this.stack = null;
    window.removeEventListener("keydown", this.onKey, true);
  }
}
