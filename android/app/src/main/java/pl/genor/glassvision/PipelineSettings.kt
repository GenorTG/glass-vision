package pl.genor.glassvision

import org.json.JSONException
import org.json.JSONObject

/** What the user changed for one pipeline; anything not set follows the server's default from `hello`. */
data class PipelineOverride(val enabled: Boolean? = null, val options: Map<String, OptionValue> = emptyMap())

/**
 * The user's pipeline choices, keyed by pipeline id. Only touched values are stored, so a pipeline the app
 * has never seen (or a changed server default) just works. Pure + immutable; persisted by [PipelineStore].
 */
data class PipelineSettings(val overrides: Map<String, PipelineOverride> = emptyMap()) {

    fun withEnabled(pipeline: String, enabled: Boolean): PipelineSettings =
        edit(pipeline) { it.copy(enabled = enabled) }

    fun withOption(pipeline: String, key: String, value: OptionValue): PipelineSettings =
        edit(pipeline) { it.copy(options = it.options + (key to value)) }

    fun isEnabled(spec: PipelineSpec): Boolean = overrides[spec.id]?.enabled ?: spec.defaultEnabled

    /** The value to show/send for [option] of [spec]: the user's (if still valid) or the server default. */
    fun optionValue(spec: PipelineSpec, option: OptionSpec): OptionValue =
        option.coerce(overrides[spec.id]?.options?.get(option.key))

    /**
     * Full `config.pipelines` for the pipelines the server announced. Only ids and option keys from [specs]
     * are included (the server answers unknown ones with `error`), values coerced to the declared types.
     */
    fun resolve(specs: List<PipelineSpec>): Map<String, PipelineConfig> =
        specs.associate { spec ->
            spec.id to PipelineConfig(
                enabled = isEnabled(spec),
                options = spec.options.associate { it.key to optionValue(spec, it) },
            )
        }

    fun toJson(): String {
        val root = JSONObject()
        for ((id, o) in overrides) {
            val p = JSONObject()
            o.enabled?.let { p.put("enabled", it) }
            if (o.options.isNotEmpty()) p.put("options", Protocol.valuesJson(o.options))
            root.put(id, p)
        }
        return root.toString()
    }

    private fun edit(pipeline: String, f: (PipelineOverride) -> PipelineOverride) =
        PipelineSettings(overrides + (pipeline to f(overrides[pipeline] ?: PipelineOverride())))

    companion object {
        /** Lenient: a corrupt blob yields no overrides (= server defaults) instead of a crash. */
        fun fromJson(json: String?): PipelineSettings {
            if (json.isNullOrBlank()) return PipelineSettings()
            return try {
                val root = JSONObject(json)
                val map = LinkedHashMap<String, PipelineOverride>()
                for (id in root.keys()) {
                    val p = root.optJSONObject(id) ?: continue
                    val opts = LinkedHashMap<String, OptionValue>()
                    p.optJSONObject("options")?.let { o ->
                        for (k in o.keys()) Protocol.optionValueOf(o.opt(k))?.let { opts[k] = it }
                    }
                    val enabled = if (p.has("enabled")) p.optBoolean("enabled") else null
                    map[id] = PipelineOverride(enabled, opts)
                }
                PipelineSettings(map)
            } catch (_: JSONException) {
                PipelineSettings()
            }
        }

        /**
         * What changed between two resolved configs: per pipeline only a flipped `enabled` and changed options.
         * [previous] null = nothing sent yet → everything.
         */
        fun delta(
            previous: Map<String, PipelineConfig>?,
            current: Map<String, PipelineConfig>,
        ): Map<String, PipelineConfig> {
            if (previous == null) return current
            val out = LinkedHashMap<String, PipelineConfig>()
            for ((id, cur) in current) {
                val prev = previous[id]
                if (prev == null) {
                    out[id] = cur
                    continue
                }
                val enabled = cur.enabled.takeIf { it != prev.enabled }
                val options = cur.options.filter { (k, v) -> prev.options[k] != v }
                if (enabled != null || options.isNotEmpty()) out[id] = PipelineConfig(enabled, options)
            }
            return out
        }
    }
}
