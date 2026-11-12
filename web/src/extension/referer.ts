/**
 * A session rule that sends `page`'s origin as the Referer of the extension's own requests to the image's host, as
 * the browser does when the page loads the image (strict-origin-when-cross-origin). Null when either URL is not
 * http(s), since then there is no Referer to imitate.
 */
export function refererRule(id: number, image: string, page: string, extensionId: string): chrome.declarativeNetRequest.Rule | null {
  let imageUrl: URL;
  let pageUrl: URL;
  try {
    imageUrl = new URL(image);
    pageUrl = new URL(page);
  } catch {
    return null;
  }
  const web = ["https:", "http:"];
  if (!web.includes(imageUrl.protocol) || !web.includes(pageUrl.protocol)) return null;
  return {
    id,
    priority: 1,
    action: {
      type: "modifyHeaders" as chrome.declarativeNetRequest.RuleActionType,
      requestHeaders: [{ header: "referer", operation: "set" as chrome.declarativeNetRequest.HeaderOperation, value: `${pageUrl.origin}/` }],
    },
    condition: {
      requestDomains: [imageUrl.hostname],
      initiatorDomains: [extensionId],
      resourceTypes: ["xmlhttprequest" as chrome.declarativeNetRequest.ResourceType],
    },
  };
}
