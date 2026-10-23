package pl.genor.glassvision

import android.content.Context
import android.content.SharedPreferences

/** Read/edit the user's pipeline choices; what the UI depends on (snapshot tests pass an in-memory one). */
interface PipelineChoices {
    fun load(): PipelineSettings

    fun update(transform: (PipelineSettings) -> PipelineSettings)

    /** Forget every user choice → server defaults. */
    fun reset() = update { PipelineSettings() }
}

/**
 * Persists [PipelineSettings] as one JSON blob in its own prefs file (so [SettingsStore] listeners don't fire
 * on pipeline edits). Main thread only; listeners run synchronously after every change.
 */
class PipelineStore(context: Context) : PipelineChoices {
    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val listeners = mutableListOf<(PipelineSettings) -> Unit>()
    private var cached: PipelineSettings? = null

    override fun load(): PipelineSettings =
        cached ?: PipelineSettings.fromJson(prefs.getString(KEY_SETTINGS, null)).also { cached = it }

    override fun update(transform: (PipelineSettings) -> PipelineSettings) {
        val old = load()
        val new = transform(old)
        if (new == old) return
        cached = new
        prefs.edit().putString(KEY_SETTINGS, new.toJson()).apply()
        listeners.toList().forEach { it(new) }
    }

    fun addListener(onChange: (PipelineSettings) -> Unit) {
        listeners += onChange
    }

    fun removeListener(onChange: (PipelineSettings) -> Unit) {
        listeners -= onChange
    }

    private companion object {
        const val PREFS_NAME = "glass_vision_pipelines"
        const val KEY_SETTINGS = "pipelines_json"
    }
}
