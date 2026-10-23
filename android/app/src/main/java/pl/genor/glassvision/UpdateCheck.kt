package pl.genor.glassvision

import org.json.JSONException
import org.json.JSONObject

/** The APK the server currently serves (`GET /app/version`). */
data class UpdateInfo(
    val applicationId: String,
    val versionCode: Int,
    val versionName: String,
    val size: Long,
    val sha256: String,
    val url: String,
)

/** Pure update logic (JVM-testable): URL derivation, parsing, version comparison. */
object UpdateCheck {
    /**
     * `ws://host:8770/ws` → `http://host:8770` (wss → https); a reverse-proxy prefix stays
     * (`wss://h/glass/ws` → `https://h/glass`). Null if it isn't a ws(s) URL.
     */
    fun httpBase(wsUrl: String): String? {
        val trimmed = wsUrl.trim()
        val scheme = when {
            trimmed.startsWith("ws://", ignoreCase = true) -> "http://"
            trimmed.startsWith("wss://", ignoreCase = true) -> "https://"
            else -> return null
        }
        val rest = trimmed.substringAfter("://").substringBefore('?').substringBefore('#')
        val authority = rest.substringBefore('/').ifEmpty { return null }
        val path = if ('/' in rest) "/" + rest.substringAfter('/') else ""
        return scheme + authority + path.trimEnd('/').removeSuffix("/ws").trimEnd('/')
    }

    fun parse(json: String): UpdateInfo? = try {
        val o = JSONObject(json)
        UpdateInfo(
            applicationId = o.getString("application_id"),
            versionCode = o.getInt("version_code"),
            versionName = o.getString("version_name"),
            size = o.getLong("size"),
            sha256 = o.getString("sha256").lowercase(),
            url = o.optString("url", "/app.apk"),
        )
    } catch (_: JSONException) {
        null
    }

    /** Only the same app, and only strictly newer: never offer a downgrade or a foreign package. */
    fun isUpdate(info: UpdateInfo, currentVersionCode: Int, applicationId: String): Boolean =
        info.applicationId == applicationId && info.versionCode > currentVersionCode
}
