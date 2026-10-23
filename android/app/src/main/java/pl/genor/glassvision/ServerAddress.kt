package pl.genor.glassvision

import java.net.URLEncoder

/**
 * What the user types as "server address" → the WebSocket URL the app connects to. Pure and JVM-testable.
 *
 * Accepts `host`, `host:port`, `http(s)://…` and `ws(s)://…`, with or without the `/ws` path:
 * - no scheme / `http` → `ws`, `https` → `wss`;
 * - no port → [DEFAULT_PORT] for plain connections; `https`/`wss` without a port mean a TLS reverse proxy on 443,
 *   so they stay portless (`wss://vision.example.com/ws`);
 * - the path keeps a reverse-proxy prefix and always ends in `/ws` (`https://h/glass` → `wss://h/glass/ws`);
 * - a `?token=…` pasted with the URL is lifted out into [Parsed.token]; other query/fragment parts are dropped.
 */
object ServerAddress {
    const val DEFAULT_PORT = 8770
    private const val WS_PATH = "/ws"
    private val HOST_NAME = Regex("^[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?(\\.[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?)*$")
    private val IPV6 = Regex("^[0-9A-Fa-f:.]+(%[A-Za-z0-9]+)?$")

    data class Parsed(val wsUrl: String, val token: String?)

    /** The normalized `ws(s)://host[:port]/…/ws` URL, or null if [input] isn't a usable address. */
    fun normalize(input: String): String? = parse(input)?.wsUrl

    fun parse(input: String): Parsed? {
        var rest = input.trim()
        if (rest.isEmpty() || rest.any(Char::isWhitespace)) return null

        val schemeEnd = rest.indexOf("://")
        val secure = if (schemeEnd >= 0) {
            when (rest.substring(0, schemeEnd).lowercase()) {
                "ws", "http" -> false
                "wss", "https" -> true
                else -> return null
            }
        } else {
            false
        }
        if (schemeEnd >= 0) rest = rest.substring(schemeEnd + 3)

        rest = rest.substringBefore('#')
        val query = rest.substringAfter('?', "")
        rest = rest.substringBefore('?')
        val authority = rest.substringBefore('/')
        val path = if ('/' in rest) "/" + rest.substringAfter('/') else ""
        if (authority.isEmpty() || '@' in authority) return null

        val (host, port) = splitHostPort(authority) ?: return null
        val portPart = when {
            port != null -> ":$port"
            secure -> "" // TLS reverse proxy on its default port
            else -> ":$DEFAULT_PORT"
        }
        val prefix = path.trimEnd('/').removeSuffix(WS_PATH).trimEnd('/')
        val url = (if (secure) "wss://" else "ws://") + host + portPart + prefix + WS_PATH
        return Parsed(url, tokenFromQuery(query))
    }

    /** `host`, `host:port`, `[v6]`, `[v6]:port` or a bare IPv6 literal → (host as it goes into a URL, port). */
    private fun splitHostPort(authority: String): Pair<String, Int?>? {
        if (authority.startsWith('[')) {
            val close = authority.indexOf(']')
            if (close < 0) return null
            val inner = authority.substring(1, close)
            if (!IPV6.matches(inner) || ':' !in inner) return null
            val after = authority.substring(close + 1)
            val port = when {
                after.isEmpty() -> null
                after.startsWith(':') -> parsePort(after.substring(1)) ?: return null
                else -> return null
            }
            return "[$inner]" to port
        }
        val colons = authority.count { it == ':' }
        if (colons > 1) {
            // Unbracketed IPv6 literal: no port possible.
            return if (IPV6.matches(authority)) "[$authority]" to null else null
        }
        val host = authority.substringBefore(':')
        if (!HOST_NAME.matches(host)) return null
        val port = if (colons == 1) parsePort(authority.substringAfter(':')) ?: return null else null
        return host.lowercase() to port
    }

    private fun parsePort(s: String): Int? = s.toIntOrNull()?.takeIf { it in 1..65535 && s.all(Char::isDigit) }

    private fun tokenFromQuery(query: String): String? = query.split('&')
        .firstOrNull { it.startsWith("token=") }
        ?.substringAfter('=')
        ?.let { runCatching { java.net.URLDecoder.decode(it, Charsets.UTF_8.name()) }.getOrNull() }
        ?.takeIf { it.isNotBlank() }
}

/**
 * Where the app talks to: the normalized WebSocket URL plus the optional access token (docs/PROTOCOL.md § Auth).
 * Empty [wsUrl] = not configured yet (first run): nothing connects. Pure and JVM-testable.
 */
data class ServerEndpoint(val wsUrl: String, val token: String = "") {
    val isConfigured: Boolean get() = wsUrl.isNotBlank()

    /** `http(s)://host:port[/prefix]` for the HTTP API (health, updates, capture files); null if not a ws URL. */
    val httpBase: String? get() = UpdateCheck.httpBase(wsUrl)

    /** The URL to open the socket with: `?token=<url-encoded>` appended when a token is set. */
    fun socketUrl(): String = withToken(wsUrl, token)

    /** `Authorization` header value for HTTP calls, or null without a token. */
    fun authorization(): String? = token.trim().takeIf { it.isNotEmpty() }?.let { "Bearer $it" }

    companion object {
        fun withToken(url: String, token: String): String {
            val t = token.trim()
            if (t.isEmpty()) return url
            // URLEncoder is form encoding: space → '+', which the server's query parser decodes back to a space.
            val sep = if ('?' in url) '&' else '?'
            return "$url${sep}token=${URLEncoder.encode(t, Charsets.UTF_8.name())}"
        }
    }
}

/** How a lost connection is treated: an auth rejection stops the retry loop and asks the user for the token. */
object ConnectionLoss {
    /** WebSocket close code the server uses for a missing/wrong token. */
    const val CLOSE_UNAUTHORIZED = 4401

    enum class Reaction { RECONNECT, AUTH_FAILED }

    /**
     * [closeCode] from a close frame, [httpStatus] from a failed handshake. Besides 4401, a 401 or 403 handshake
     * response counts: an ASGI server that closes before `accept()` answers the upgrade with HTTP 403.
     */
    fun reaction(closeCode: Int?, httpStatus: Int?): Reaction = when {
        closeCode == CLOSE_UNAUTHORIZED -> Reaction.AUTH_FAILED
        httpStatus == 401 || httpStatus == 403 -> Reaction.AUTH_FAILED
        else -> Reaction.RECONNECT
    }
}
