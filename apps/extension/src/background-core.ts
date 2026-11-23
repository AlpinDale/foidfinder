// Shared background orchestration for Chrome (service worker + offscreen document) and Firefox (event page):
// adds "Identify with foidfinder" to the image context menu, fetches/locates/captures the image, and shows cards.

import { ALL_SITES, type CardState, type IdentifyReply, type ImageSource, type LocateReply, type ReadReply, type ToBackground, type ToContent } from "./messages";
import { refererRule } from "./referer";

const MENU = "foidfinder-identify";

export interface BackgroundHandle {
  notifyProgress: (loaded: number, total: number) => void;
  hasInFlight: () => boolean;
}

export function createBackground(identifyImage: (image: ImageSource) => Promise<IdentifyReply>): BackgroundHandle {
  /** Cards waiting for the model, updated with download progress. */
  const waiting = new Map<number, (card: CardState) => void>();
  let inFlight = 0;
  // Request ids double as session rule ids; a random start avoids rules left over by a previous background instance.
  let nextId = 1 + Math.floor(Math.random() * 1e9);

  async function clearRules() {
    const rules = await chrome.declarativeNetRequest.getSessionRules();
    await chrome.declarativeNetRequest.updateSessionRules({ removeRuleIds: rules.map((r) => r.id) });
  }

  chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({ id: MENU, title: "Identify with foidfinder", contexts: ["image"] });
    void clearRules();
  });
  chrome.runtime.onStartup.addListener(() => void clearRules());

  chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === MENU && tab?.id !== undefined && info.srcUrl) void identify(tab.id, tab.windowId, info);
  });

  chrome.runtime.onMessage.addListener((msg: ToBackground) => {
    if (msg?.target === "background" && msg.type === "open-options") void chrome.runtime.openOptionsPage();
    return false;
  });

  async function infer(image: ImageSource): Promise<IdentifyReply> {
    inFlight += 1;
    try {
      return await identifyImage(image);
    } finally {
      inFlight -= 1;
    }
  }

  function toTab<T>(tabId: number, frameId: number, msg: ToContent): Promise<T | undefined> {
    return chrome.tabs.sendMessage<ToContent, T>(tabId, msg, { frameId }).catch(() => undefined);
  }

  /**
   * Downloads the image. With access to all sites it goes out as the page's request did, with the page's origin as
   * Referer, which hotlink protection checks; without, it succeeds only where the image host allows cross-origin reads.
   */
  async function fromUrl(id: number, url: string, page: string | undefined, allSites: boolean): Promise<IdentifyReply> {
    // Use the runtime URL's hostname rather than runtime.id so Firefox's internal moz-extension:// UUID is matched.
    const initiator = new URL(chrome.runtime.getURL("")).hostname;
    const rule = page && allSites ? refererRule(id, url, page, initiator) : null;
    if (rule) await chrome.declarativeNetRequest.updateSessionRules({ removeRuleIds: [id], addRules: [rule] });
    try {
      return await infer({ kind: "url", url });
    } finally {
      if (rule) await chrome.declarativeNetRequest.updateSessionRules({ removeRuleIds: [id] });
    }
  }

  /**
   * The image's sources, best first: its URL; then the page reading it (blob: URLs and images the page may read);
   * then, for images in the top frame, a crop of a screenshot of the visible tab.
   */
  async function run(id: number, tabId: number, windowId: number, info: chrome.contextMenus.OnClickData, allSites: boolean) {
    const src = info.srcUrl ?? "";
    const frameId = info.frameId ?? 0;
    let reply: IdentifyReply | null = null;
    if (/^(https?|data):/.test(src)) {
      reply = await fromUrl(id, src, info.frameUrl ?? info.pageUrl, allSites);
      if (reply.ok || reply.error.code !== "fetch_failed") return { reply, screenshot: false };
    }
    const dataUrl = await toTab<ReadReply>(tabId, frameId, { type: "ff:read", url: src });
    if (dataUrl) {
      reply = await infer({ kind: "url", url: dataUrl });
      if (reply.ok || reply.error.code !== "fetch_failed") return { reply, screenshot: false };
    }
    if (frameId === 0) {
      const where = await toTab<LocateReply>(tabId, 0, { type: "ff:locate", url: src });
      if (where) {
        const shot = await chrome.tabs.captureVisibleTab(windowId, { format: "png" });
        const s = where.dpr;
        const rect = {
          x: Math.round(where.rect.x * s),
          y: Math.round(where.rect.y * s),
          width: Math.round(where.rect.width * s),
          height: Math.round(where.rect.height * s),
        };
        return { reply: await infer({ kind: "screenshot", dataUrl: shot, rect }), screenshot: true };
      }
    }
    reply ??= { ok: false, error: { code: "fetch_failed", message: "The image could not be read" } };
    return { reply, screenshot: false };
  }

  async function identify(tabId: number, windowId: number, info: chrome.contextMenus.OnClickData) {
    const id = nextId++;
    const frameId = info.frameId ?? 0;
    const image = info.srcUrl ?? "";
    const topScript = chrome.scripting.executeScript({ target: { tabId, frameIds: [0] }, files: ["content.js"] });
    // The image's own frame may be asked to read it; a frame from another site needs access to all sites.
    const subScript =
      frameId !== 0
        ? chrome.scripting.executeScript({ target: { tabId, frameIds: [frameId] }, files: ["content.js"] }).catch(() => {})
        : Promise.resolve();
    const perms = chrome.permissions.contains(ALL_SITES);
    try {
      // The top frame shows the cards (activeTab allows it after the menu click).
      await topScript;
    } catch {
      return; // pages extensions may not script, such as browser internal or store pages
    }
    const [, allSites] = await Promise.all([subScript, perms]);
    const show = (card: CardState) => void toTab(tabId, 0, { type: "ff:card", id, image, card });
    show({ state: "loading" });
    waiting.set(id, show);
    try {
      const { reply, screenshot } = await run(id, tabId, windowId, info, allSites);
      const upgrade = !allSites && (screenshot || (!reply.ok && reply.error.code === "fetch_failed"));
      show(
        reply.ok
          ? { state: "result", result: reply.result, screenshot, upgrade }
          : { state: "error", message: reply.error.message, upgrade },
      );
    } catch (err) {
      show({ state: "error", message: err instanceof Error ? err.message : String(err), upgrade: false });
    } finally {
      waiting.delete(id);
    }
  }

  return {
    notifyProgress: (loaded, total) => waiting.forEach((show) => show({ state: "download", loaded, total })),
    hasInFlight: () => inFlight > 0,
  };
}
