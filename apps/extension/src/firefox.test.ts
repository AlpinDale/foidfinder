import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const runnerMocks = vi.hoisted(() => ({
  identify: vi.fn(),
  isCached: vi.fn<() => Promise<boolean>>(),
  warm: vi.fn<() => Promise<boolean>>(),
  dispose: vi.fn(),
  lastCallbacks: null as { onProgress: (l: number, t: number) => void; onIdle: () => void } | null,
}));

vi.mock("./runner", () => ({
  createRunner: (callbacks: { onProgress: (l: number, t: number) => void; onIdle: () => void }) => {
    runnerMocks.lastCallbacks = callbacks;
    return {
      identify: runnerMocks.identify,
      isCached: runnerMocks.isCached,
      warm: runnerMocks.warm,
      dispose: runnerMocks.dispose,
    };
  },
}));

type StorageListener = (changes: Record<string, { oldValue?: unknown; newValue?: unknown }>, area: string) => void;
type ClickListener = (info: chrome.contextMenus.OnClickData, tab?: chrome.tabs.Tab) => void;

function stubChrome(stored: Record<string, unknown> = {}) {
  let onStorageChanged: StorageListener | null = null;
  let onContextClick: ClickListener | null = null;
  const getPlatformInfo = vi.fn().mockResolvedValue({ os: "mac", arch: "arm" });
  const updateSessionRules = vi.fn().mockResolvedValue(undefined);
  const sendMessage = vi.fn().mockResolvedValue(undefined);

  vi.stubGlobal("chrome", {
    runtime: {
      id: "extension@foidfinder.net",
      getURL: (p: string) => `moz-extension://1111-2222-3333/${p}`,
      getPlatformInfo,
      openOptionsPage: vi.fn(),
      onInstalled: { addListener: vi.fn() },
      onStartup: { addListener: vi.fn() },
      onMessage: { addListener: vi.fn() },
    },
    contextMenus: {
      create: vi.fn(),
      onClicked: {
        addListener: (fn: ClickListener) => {
          onContextClick = fn;
        },
      },
    },
    declarativeNetRequest: {
      getSessionRules: vi.fn().mockResolvedValue([]),
      updateSessionRules,
    },
    scripting: {
      executeScript: vi.fn().mockResolvedValue([]),
    },
    permissions: {
      contains: vi.fn().mockResolvedValue(true),
    },
    tabs: {
      sendMessage,
      captureVisibleTab: vi.fn(),
    },
    storage: {
      local: {
        get: vi.fn().mockResolvedValue(stored),
      },
      onChanged: {
        addListener: (fn: StorageListener) => {
          onStorageChanged = fn;
        },
      },
    },
  });

  return {
    getPlatformInfo,
    updateSessionRules,
    triggerStorage: (changes: Record<string, { oldValue?: unknown; newValue?: unknown }>, area = "local") =>
      onStorageChanged?.(changes, area),
    triggerClick: (info: chrome.contextMenus.OnClickData, tab: chrome.tabs.Tab) => onContextClick?.(info, tab),
  };
}

beforeEach(() => {
  vi.resetModules();
  vi.useFakeTimers();
  runnerMocks.identify.mockReset();
  runnerMocks.isCached.mockReset().mockResolvedValue(true);
  runnerMocks.warm.mockReset().mockResolvedValue(true);
  runnerMocks.dispose.mockReset();
  runnerMocks.lastCallbacks = null;
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Firefox background event page", () => {
  it("does not keepalive or pre-warm by default, and disposes on idle", async () => {
    const { getPlatformInfo } = stubChrome({});
    await import("./background.firefox");
    await vi.advanceTimersByTimeAsync(60_000);

    expect(runnerMocks.warm).not.toHaveBeenCalled();
    expect(getPlatformInfo).not.toHaveBeenCalled();

    runnerMocks.lastCallbacks?.onIdle();
    expect(runnerMocks.dispose).toHaveBeenCalled();
  });

  it("keeps the event page alive during an active identification/download even when keepAlive is off, then stops", async () => {
    const { getPlatformInfo, triggerClick } = stubChrome({});
    const pending = Promise.withResolvers<unknown>();
    runnerMocks.identify.mockReturnValue(pending.promise);
    await import("./background.firefox");

    triggerClick(
      {
        menuItemId: "foidfinder-identify",
        editable: false,
        srcUrl: "https://example.com/a.png",
        pageUrl: "https://example.com/",
      },
      { id: 1, windowId: 1 } as chrome.tabs.Tab,
    );
    // Heartbeat runs while the download/identification is in flight (every 15s + initial ping).
    await vi.advanceTimersByTimeAsync(45_000);
    const pingsDuringDownload = getPlatformInfo.mock.calls.length;
    expect(pingsDuringDownload).toBeGreaterThanOrEqual(3);

    // Once finished, heartbeat stops because keepAlive is off by default.
    pending.resolve({
      ok: true,
      result: { rating: null, tags: {}, candidates: {}, image: { width: 10, height: 10 }, timings_ms: { decode: 1, inference: 1, total: 2 } },
    });
    await vi.advanceTimersByTimeAsync(45_000);
    expect(getPlatformInfo.mock.calls.length).toBe(pingsDuringDownload);
  });

  it("starts keepalive and warms when toggled on, and stops/disposes when toggled off", async () => {
    const { getPlatformInfo, triggerStorage } = stubChrome({});
    await import("./background.firefox");
    await Promise.resolve();

    triggerStorage({ keepAlive: { oldValue: false, newValue: true } });
    await vi.advanceTimersByTimeAsync(15_000);

    expect(runnerMocks.warm).toHaveBeenCalledTimes(1);
    expect(getPlatformInfo.mock.calls.length).toBeGreaterThanOrEqual(2);
    const pingsBeforeOff = getPlatformInfo.mock.calls.length;

    // Idle callback does not dispose while keepAlive is enabled.
    runnerMocks.dispose.mockClear();
    runnerMocks.lastCallbacks?.onIdle();
    expect(runnerMocks.dispose).not.toHaveBeenCalled();

    triggerStorage({ keepAlive: { oldValue: true, newValue: false } });
    await vi.advanceTimersByTimeAsync(40_000);

    expect(runnerMocks.dispose).toHaveBeenCalledTimes(1);
    expect(getPlatformInfo.mock.calls.length).toBe(pingsBeforeOff);
  });

  it("uses the moz-extension:// URL hostname as DNR initiatorDomains instead of runtime.id", async () => {
    const { updateSessionRules, triggerClick } = stubChrome({});
    runnerMocks.identify.mockResolvedValue({
      ok: true,
      result: { rating: null, tags: {}, candidates: {}, image: { width: 10, height: 10 }, timings_ms: { decode: 1, inference: 1, total: 2 } },
    });
    await import("./background.firefox");

    triggerClick(
      {
        menuItemId: "foidfinder-identify",
        editable: false,
        srcUrl: "https://i.pximg.net/img/1.png",
        pageUrl: "https://www.pixiv.net/artworks/1",
      },
      { id: 1, windowId: 1 } as chrome.tabs.Tab,
    );
    await vi.advanceTimersByTimeAsync(0);

    const addCall = updateSessionRules.mock.calls.find(([arg]) => Array.isArray((arg as { addRules?: unknown[] }).addRules));
    expect(addCall).toBeDefined();
    const rule = (addCall?.[0] as { addRules: chrome.declarativeNetRequest.Rule[] }).addRules[0];
    expect(rule?.condition.initiatorDomains).toEqual(["1111-2222-3333"]);
  });
});
