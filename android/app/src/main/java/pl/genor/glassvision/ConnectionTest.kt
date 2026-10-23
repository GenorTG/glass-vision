package pl.genor.glassvision

import android.os.Handler
import android.os.Looper
import org.json.JSONException
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.Executors

/** `GET /health` → `{"ok": true, "device": "...", "version": "..."}`. */
data class HealthInfo(val device: String, val version: String)

/** Outcome of the "Test connection" button. */
sealed interface ConnectionTestResult {
    /** The server answered as a glass-vision server and accepted the token (or needs none). */
    data class Ok(val info: HealthInfo) : ConnectionTestResult

    /** Reachable, but the protected API said 401/403: [tokenGiven] false = the server needs one. */
    data class WrongToken(val info: HealthInfo, val tokenGiven: Boolean) : ConnectionTestResult

    /** Something answered, but not with a glass-vision `/health`. */
    data class NotGlassVision(val detail: String) : ConnectionTestResult

    data class Unreachable(val detail: String) : ConnectionTestResult

    data object InvalidAddress : ConnectionTestResult
}

/** Pure parts of the connection test (JVM-testable). */
object ConnectionTests {
    fun parseHealth(json: String): HealthInfo? = try {
        val o = JSONObject(json)
        if (!o.optBoolean("ok", false)) null else HealthInfo(o.optString("device"), o.optString("version"))
    } catch (_: JSONException) {
        null
    }

    /**
     * `/health` is open even on token-protected servers, so the token is checked against a protected endpoint
     * (`/app/version`): [authStatus] 401/403 = rejected; anything else (200, 404 without an APK…) = accepted.
     */
    fun classify(healthStatus: Int, health: HealthInfo?, authStatus: Int?, tokenGiven: Boolean): ConnectionTestResult =
        when {
            health == null -> ConnectionTestResult.NotGlassVision("HTTP $healthStatus")
            authStatus == 401 || authStatus == 403 -> ConnectionTestResult.WrongToken(health, tokenGiven)
            else -> ConnectionTestResult.Ok(health)
        }
}

/** Runs the test on a worker thread; [test]'s callback runs on the main thread. */
class ConnectionTester {
    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "gv-conn-test").apply { isDaemon = true } }

    fun test(endpoint: ServerEndpoint, onResult: (ConnectionTestResult) -> Unit) {
        worker.execute {
            val result = run(endpoint)
            main.post { onResult(result) }
        }
    }

    fun shutdown() = worker.shutdownNow()

    private fun run(endpoint: ServerEndpoint): ConnectionTestResult {
        val healthRequest = ServerHttp.get(endpoint, "/health") ?: return ConnectionTestResult.InvalidAddress
        val (status, body) = try {
            ServerHttp.client.newCall(healthRequest).execute().use { r ->
                r.code to (if (r.isSuccessful) r.body?.string().orEmpty().take(MAX_BODY) else "")
            }
        } catch (e: IOException) {
            return ConnectionTestResult.Unreachable(e.message ?: e.javaClass.simpleName)
        }
        val health = ConnectionTests.parseHealth(body)
        val authStatus = if (health == null) {
            null
        } else {
            ServerHttp.get(endpoint, "/app/version")?.let { request ->
                try {
                    ServerHttp.client.newCall(request).execute().use { it.code }
                } catch (_: IOException) {
                    null
                }
            }
        }
        return ConnectionTests.classify(status, health, authStatus, endpoint.token.isNotBlank())
    }

    private companion object {
        const val MAX_BODY = 16 * 1024
    }
}
