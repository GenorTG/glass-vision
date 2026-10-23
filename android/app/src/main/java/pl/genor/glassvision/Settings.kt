package pl.genor.glassvision

import android.content.Context
import android.content.SharedPreferences

/** Immutable snapshot of user-tunable streaming settings. */
data class StreamSettings(
    /** Normalized WebSocket URL ([ServerAddress]); empty = not configured yet (first-run sheet, no connecting). */
    val serverUrl: String = "",
    /** Optional access token (docs/PROTOCOL.md § Auth); empty = the server runs without GV_TOKEN. */
    val token: String = "",
    val targetFps: Int = 10,
    val inFlightWindow: Int = 2,
    val sendLongSide: Int = 960,
    val jpegQuality: Int = 70,
    val confidence: Float = 0.35f,
    val modelImgsz: Int = 640,
    /** Draw server-smoothed tracks (true) or raw per-frame detections (false). */
    val smoothing: Boolean = true,
    /** How long the server keeps showing a track after it was last detected. */
    val holdMs: Int = 300,
    /** Box glide time constant (client-side animation between results); 0 = off. */
    val glideMs: Int = 70,
    /** Streaming camera; the flip button and Settings → Camera both write it. */
    val cameraFacing: CameraFacing = CameraFacing.BACK,
) {
    /** Clamp every field into its documented range so bad prefs can never reach the pipeline. */
    fun sanitized(): StreamSettings = copy(
        serverUrl = serverUrl.trim(),
        token = token.trim(),
        targetFps = targetFps.coerceIn(FPS_RANGE),
        inFlightWindow = inFlightWindow.coerceIn(WINDOW_RANGE),
        sendLongSide = sendLongSide.takeIf { it in SEND_SIZES } ?: 960,
        jpegQuality = jpegQuality.coerceIn(QUALITY_RANGE),
        confidence = confidence.coerceIn(CONF_MIN, CONF_MAX),
        modelImgsz = modelImgsz.takeIf { it in IMGSZ_OPTIONS } ?: 640,
        holdMs = (holdMs.coerceIn(HOLD_RANGE) / HOLD_STEP) * HOLD_STEP,
        glideMs = glideMs.takeIf { it in GLIDE_OPTIONS } ?: 70,
    )

    val endpoint: ServerEndpoint get() = ServerEndpoint(serverUrl, token)

    companion object {
        val FPS_RANGE = 1..30
        val WINDOW_RANGE = 1..4
        val QUALITY_RANGE = 30..95
        val SEND_SIZES = listOf(640, 960, 1280)
        val IMGSZ_OPTIONS = listOf(320, 480, 640, 960, 1280)
        const val CONF_MIN = 0.10f // server detects down to 0.1 for tracking; nothing exists below it
        const val CONF_MAX = 0.95f
        const val CONF_STEP = 0.05f
        val HOLD_RANGE = 0..1000
        const val HOLD_STEP = 50
        /** Box glide: off / smooth / extra smooth (time constant in ms). */
        val GLIDE_OPTIONS = listOf(0, 70, 140)
    }
}

/** Read/edit [StreamSettings]; what the UI depends on (snapshot tests pass an in-memory one). */
interface StreamSettingsEditor {
    fun load(): StreamSettings

    fun update(transform: (StreamSettings) -> StreamSettings)

    /** Defaults for everything except where to connect and which camera streams. */
    fun resetToDefaults() =
        update { StreamSettings(serverUrl = it.serverUrl, token = it.token, cameraFacing = it.cameraFacing) }
}

/** SharedPreferences-backed persistence for [StreamSettings]. */
class SettingsStore(context: Context) : StreamSettingsEditor {
    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    // SharedPreferences holds listeners weakly; keep strong refs here.
    private val listeners = mutableMapOf<(StreamSettings) -> Unit, SharedPreferences.OnSharedPreferenceChangeListener>()

    override fun load(): StreamSettings {
        val d = StreamSettings()
        return StreamSettings(
            serverUrl = prefs.getString(KEY_URL, d.serverUrl) ?: d.serverUrl,
            token = prefs.getString(KEY_TOKEN, d.token) ?: d.token,
            targetFps = prefs.getInt(KEY_FPS, d.targetFps),
            inFlightWindow = prefs.getInt(KEY_WINDOW, d.inFlightWindow),
            sendLongSide = prefs.getInt(KEY_SEND_SIZE, d.sendLongSide),
            jpegQuality = prefs.getInt(KEY_QUALITY, d.jpegQuality),
            confidence = prefs.getFloat(KEY_CONF, d.confidence),
            modelImgsz = prefs.getInt(KEY_IMGSZ, d.modelImgsz),
            smoothing = prefs.getBoolean(KEY_SMOOTHING, d.smoothing),
            holdMs = prefs.getInt(KEY_HOLD_MS, d.holdMs),
            glideMs = prefs.getInt(KEY_GLIDE_MS, d.glideMs),
            cameraFacing = CameraFacing.fromPref(prefs.getString(KEY_CAMERA, null)),
        ).sanitized()
    }

    fun save(settings: StreamSettings) {
        val s = settings.sanitized()
        prefs.edit()
            .putString(KEY_URL, s.serverUrl)
            // Plain app-private prefs, not the Keystore: the token is a convenience secret that keeps strangers
            // off a self-hosted server, not a credential worth hardware-backed storage (allowBackup is off).
            .putString(KEY_TOKEN, s.token)
            .putInt(KEY_FPS, s.targetFps)
            .putInt(KEY_WINDOW, s.inFlightWindow)
            .putInt(KEY_SEND_SIZE, s.sendLongSide)
            .putInt(KEY_QUALITY, s.jpegQuality)
            .putFloat(KEY_CONF, s.confidence)
            .putInt(KEY_IMGSZ, s.modelImgsz)
            .putBoolean(KEY_SMOOTHING, s.smoothing)
            .putInt(KEY_HOLD_MS, s.holdMs)
            .putInt(KEY_GLIDE_MS, s.glideMs)
            .putString(KEY_CAMERA, s.cameraFacing.prefValue)
            .apply()
    }

    override fun update(transform: (StreamSettings) -> StreamSettings) = save(transform(load()))

    /** [onChange] runs on the main thread with the freshly loaded settings. */
    fun addListener(onChange: (StreamSettings) -> Unit) {
        val l = SharedPreferences.OnSharedPreferenceChangeListener { _, _ -> onChange(load()) }
        listeners[onChange] = l
        prefs.registerOnSharedPreferenceChangeListener(l)
    }

    fun removeListener(onChange: (StreamSettings) -> Unit) {
        listeners.remove(onChange)?.let(prefs::unregisterOnSharedPreferenceChangeListener)
    }

    private companion object {
        const val PREFS_NAME = "glass_vision"
        const val KEY_URL = "server_url"
        const val KEY_TOKEN = "server_token"
        const val KEY_FPS = "target_fps"
        const val KEY_WINDOW = "in_flight_window"
        const val KEY_SEND_SIZE = "send_long_side"
        const val KEY_QUALITY = "jpeg_quality"
        const val KEY_CONF = "confidence"
        const val KEY_IMGSZ = "model_imgsz"
        const val KEY_SMOOTHING = "smoothing"
        const val KEY_HOLD_MS = "hold_ms"
        const val KEY_GLIDE_MS = "glide_ms"
        const val KEY_CAMERA = "camera_facing"
    }
}
