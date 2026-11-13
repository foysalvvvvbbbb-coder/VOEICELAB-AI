/**
 * Safe API JSON client with:
 * 1. Same-origin relative URL normalization
 * 2. credentials: 'include' for cross-site iframe & top-level Partitioned cookies
 * 3. Pre-mutation GET /api/auth/session warmup (prevents Nginx Lua Auth Bridge cache-miss on POST multipart uploads)
 * 4. Hidden iframe Auth Bridge auto-recovery if __SECURE-aistudio_auth_token expired
 * 5. Dual-transport HMAC-signed session token (HttpOnly Cookie + X-VoiceLab-Session header)
 */
function normalizeApiBaseUrl(): string {
  const raw =
    typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL
      ? String(import.meta.env.VITE_API_BASE_URL).trim()
      : '';

  if (
    !raw ||
    raw === '/' ||
    raw === '""' ||
    raw === "''" ||
    /^(same-origin|relative|none|null|undefined)$/i.test(raw) ||
    /^(MY_|YOUR_|REPLACE_)/i.test(raw)
  ) {
    return '';
  }

  const stripped = raw.replace(/\/+$/, '');

  if (
    typeof window !== 'undefined' &&
    window.location.hostname !== 'localhost' &&
    window.location.hostname !== '127.0.0.1' &&
    /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(stripped)
  ) {
    return '';
  }

  return stripped;
}

const RAW_API_BASE = normalizeApiBaseUrl();

// In-memory session state (never logged or exposed)
let cachedSessionToken: string | null = null;
let lastSessionVerifiedAtMs = 0;
let inflightSessionHandshake: Promise<boolean> | null = null;

// Keep session & Nginx Lua auth_token_cache (300s TTL) warm by refreshing if > 180s old
const SESSION_WARM_TTL_MS = 180_000;

export function resolveApiUrl(url: string): string {
  if (!RAW_API_BASE || /^https?:\/\//i.test(url)) {
    return url;
  }
  return url.startsWith('/') ? `${RAW_API_BASE}${url}` : `${RAW_API_BASE}/${url}`;
}

/**
 * Mounts a hidden 1x1 document iframe to /api/auth/session so the browser can complete
 * the Google AI Studio Auth Bridge 302 redirect flow and re-mint the Partitioned
 * __SECURE-aistudio_auth_token cookie when expired or cleared.
 */
async function recoverAuthBridgeSessionViaHiddenIframe(): Promise<boolean> {
  if (typeof document === 'undefined' || !document.body) {
    return false;
  }
  return new Promise<boolean>((resolve) => {
    const iframe = document.createElement('iframe');
    iframe.style.position = 'fixed';
    iframe.style.width = '1px';
    iframe.style.height = '1px';
    iframe.style.opacity = '0';
    iframe.style.pointerEvents = 'none';
    iframe.style.border = '0';
    iframe.setAttribute('aria-hidden', 'true');

    let settled = false;
    const finish = (ok: boolean) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try {
        iframe.remove();
      } catch {
        // Ignore removal errors
      }
      resolve(ok);
    };

    const timer = window.setTimeout(() => finish(false), 4500);
    iframe.onload = () => {
      // Allow a brief tick for Set-Cookie headers from the iframe navigation to commit
      window.setTimeout(() => finish(true), 250);
    };
    iframe.onerror = () => finish(false);

    iframe.src = `${resolveApiUrl('/api/auth/session')}?__reauth=${Date.now()}`;
    document.body.appendChild(iframe);
  });
}

/**
 * Performs a lightweight GET /api/auth/session handshake.
 * Ensures both the Express session cookie (voicelab_sid) and the AI Studio Nginx
 * Auth Bridge Lua cache (auth_token_cache, 300s TTL) are warm before a POST upload begins.
 */
export async function ensureSessionInitialized(force = false): Promise<boolean> {
  const now = Date.now();
  if (!force && cachedSessionToken && now - lastSessionVerifiedAtMs < SESSION_WARM_TTL_MS) {
    return true;
  }
  if (inflightSessionHandshake) {
    return inflightSessionHandshake;
  }

  inflightSessionHandshake = (async () => {
    try {
      const sessionUrl = resolveApiUrl('/api/auth/session');
      const headers = new Headers({
        Accept: 'application/json',
        'X-Requested-With': 'VoiceLab-Web-Client'
      });
      if (cachedSessionToken) {
        headers.set('X-VoiceLab-Session', cachedSessionToken);
      }

      const res = await fetch(sessionUrl, {
        method: 'GET',
        headers,
        credentials: 'include',
        cache: 'no-store'
      });

      const contentType = (res.headers.get('content-type') || '').toLowerCase();
      const rawText = await res.text();
      const trimmed = rawText.trimStart();

      const isHtml =
        contentType.includes('text/html') ||
        trimmed.toLowerCase().startsWith('<!doctype') ||
        trimmed.toLowerCase().startsWith('<html');

      if (isHtml || res.redirected) {
        // Attempt silent hidden-iframe recovery for AI Studio Auth Bridge cookie expiration
        const recovered = await recoverAuthBridgeSessionViaHiddenIframe();
        if (recovered) {
          const retryRes = await fetch(sessionUrl, {
            method: 'GET',
            headers,
            credentials: 'include',
            cache: 'no-store'
          });
          const retryText = (await retryRes.text()).trimStart();
          if (retryRes.ok && retryText.startsWith('{')) {
            const parsedRetry = JSON.parse(retryText);
            const headerTok = retryRes.headers.get('x-voicelab-session-token');
            if (parsedRetry?.sessionToken || headerTok) {
              cachedSessionToken = parsedRetry.sessionToken || headerTok;
            }
            lastSessionVerifiedAtMs = Date.now();
            return true;
          }
        }
        return false;
      }

      if (res.ok && trimmed.startsWith('{')) {
        const parsed = JSON.parse(trimmed);
        const headerTok = res.headers.get('x-voicelab-session-token');
        if (parsed?.sessionToken || headerTok) {
          cachedSessionToken = parsed.sessionToken || headerTok;
        }
        lastSessionVerifiedAtMs = Date.now();
        return true;
      }
      return false;
    } catch {
      return false;
    } finally {
      inflightSessionHandshake = null;
    }
  })();

  return inflightSessionHandshake;
}

export async function fetchApiJson<T = any>(
  url: string,
  options: RequestInit = {},
  maxWarmupRetries = 3
): Promise<T> {
  const targetUrl = resolveApiUrl(url);
  const method = String(options.method || 'GET').toUpperCase();
  const isMutation = method === 'POST' || method === 'PUT' || method === 'PATCH' || method === 'DELETE';

  // Always warm the GET session & Nginx auth_token_cache before any POST/DELETE request
  if (isMutation && !targetUrl.endsWith('/api/auth/session')) {
    await ensureSessionInitialized(false);
  }

  const headers = new Headers(options.headers || {});
  if (!headers.has('Accept')) {
    headers.set('Accept', 'application/json');
  }
  if (!headers.has('X-Requested-With')) {
    headers.set('X-Requested-With', 'VoiceLab-Web-Client');
  }

  // Never manually set Content-Type on FormData; browser must generate the multipart boundary
  if (typeof FormData !== 'undefined' && options.body instanceof FormData) {
    headers.delete('Content-Type');
  }

  let attempt = 0;
  let authRecoveryAttempted = false;

  while (true) {
    if (cachedSessionToken) {
      headers.set('X-VoiceLab-Session', cachedSessionToken);
    }

    const response = await fetch(targetUrl, {
      ...options,
      headers,
      credentials: 'include'
    });

    const refreshedTok = response.headers.get('x-voicelab-session-token');
    if (refreshedTok) {
      cachedSessionToken = refreshedTok;
      lastSessionVerifiedAtMs = Date.now();
    }

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
        if (!authRecoveryAttempted) {
          authRecoveryAttempted = true;
          const recovered = await recoverAuthBridgeSessionViaHiddenIframe();
          if (recovered) {
            await ensureSessionInitialized(true);
            continue;
          }
        }
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

    if (response.status === 401 && !authRecoveryAttempted) {
      authRecoveryAttempted = true;
      const reinitialized = await ensureSessionInitialized(true);
      if (reinitialized) {
        continue;
      }
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

    lastSessionVerifiedAtMs = Date.now();
    return parsed as T;
  }
}
