package pl.genor.glassvision

import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.concurrent.TimeUnit

/**
 * HTTP side of a [ServerEndpoint]: every call goes to its [ServerEndpoint.httpBase] and carries the access token
 * as `Authorization: Bearer …` (docs/PROTOCOL.md § Auth). One shared OkHttp client for health checks, updates
 * and capture files.
 */
object ServerHttp {
    val client: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .build()
    }

    /** GET [path] (server-relative, e.g. `/app/version`) on [endpoint]; null if the endpoint has no HTTP base. */
    fun get(endpoint: ServerEndpoint, path: String): Request? {
        val base = endpoint.httpBase ?: return null
        val url = base + (if (path.startsWith('/')) path else "/$path")
        return try {
            Request.Builder().url(url).apply { endpoint.authorization()?.let { header("Authorization", it) } }.build()
        } catch (_: IllegalArgumentException) {
            null
        }
    }
}
