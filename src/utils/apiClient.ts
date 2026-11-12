/**
 * Safe API JSON client that verifies HTTP status and Content-Type before parsing JSON,
 * automatically retries transient container warmup HTML responses (Nginx /warmup.html on 502/503/504),
 * and surfaces a clear diagnostic message if an HTML document is ever returned instead of JSON.
 */
export async function fetchApiJson<T = any>(
  url: string,
  options: RequestInit = {},
  maxWarmupRetries = 3
): Promise<T> {
  const headers = new Headers(options.headers || {});
  if (!headers.has('Accept')) {
    headers.set('Accept', 'application/json');
  }

  let attempt = 0;
  while (true) {
    const response = await fetch(url, {
      ...options,
      headers,
      credentials: 'same-origin'
    });

    const contentType = (response.headers.get('content-type') || '').toLowerCase();
    const rawText = await response.text();
    const trimmed = rawText.trimStart();

    const isHtmlPayload =
      contentType.includes('text/html') ||
      trimmed.toLowerCase().startsWith('<!doctype') ||
      trimmed.toLowerCase().startsWith('<html');

    if (isHtmlPayload) {
      const isWarmupPage =
        trimmed.includes('Starting Server') ||
        trimmed.includes('warmup') ||
        response.status === 502 ||
        response.status === 503 ||
        response.status === 504;

      if (isWarmupPage && attempt < maxWarmupRetries) {
        attempt++;
        await new Promise((resolve) => setTimeout(resolve, 1200 * attempt));
        continue;
      }

      const isCookieCheckOrAuth =
        response.redirected ||
        response.url.includes('__cookie_check') ||
        response.url.includes('applet-auth-bridge') ||
        trimmed.includes('cookie_check');

      if (isCookieCheckOrAuth) {
        throw new Error(
          `Session authentication check intercepted the API request to ${url}. Please refresh the page and try again.`
        );
      }

      const titleMatch = trimmed.match(/<title[^>]*>([^<]+)<\/title>/i);
      const htmlTitle = titleMatch ? titleMatch[1].trim() : `HTTP ${response.status}`;
      throw new Error(
        `API endpoint ${url} returned an HTML document (${htmlTitle}) instead of JSON. The server may be restarting—please try again in a moment.`
      );
    }

    if (!trimmed) {
      throw new Error(
        `API endpoint ${url} returned an empty response body (HTTP ${response.status}).`
      );
    }

    let parsed: any;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      throw new Error(
        `Invalid JSON response from ${url} (HTTP ${response.status}, Content-Type: ${
          contentType || 'unknown'
        }).`
      );
    }

    if (!response.ok) {
      throw new Error(
        parsed?.error || parsed?.message || `API request to ${url} failed with HTTP ${response.status}.`
      );
    }

    return parsed as T;
  }
}
