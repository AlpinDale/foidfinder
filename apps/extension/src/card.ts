// The result cards the content script shows at the bottom-right of a page, in a closed shadow root so neither the
// page's styles nor its scripts reach them. Text goes in through textContent only.

import { danbooruUrl, faqUrlForError, formatBytes, formatDuration, percent, splitTag, type Tag } from "@foidfinder/engine";

import cardCss from "./card.css?inline";
import type { CardState } from "./messages";

const MAX_CARDS = 3;

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

function helpLink(message: string): HTMLElement {
  const link = el("a", "help", "Troubleshooting (FAQ)");
  link.href = faqUrlForError(message);
  link.target = "_blank";
  link.rel = "noreferrer";
  return link;
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
      return [
        el("p", "error", `Couldn't identify this image. ${card.message}`),
        ...(card.upgrade ? [upgradeButton(onUpgrade)] : []),
        helpLink(card.message),
      ];
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
    style.textContent = cardCss;
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
