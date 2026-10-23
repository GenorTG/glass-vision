package pl.genor.glassvision

import android.content.Context
import org.json.JSONException
import org.json.JSONObject

/**
 * Last argument values used per pipeline action (e.g. the name typed for Learn face), so a repeated capture
 * starts pre-filled. Values are re-coerced against the action's current spec on load. Main thread only.
 */
class ArgMemory(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun load(pipeline: String, action: ActionSpec): Map<String, OptionValue> {
        val stored = try {
            JSONObject(prefs.getString(key(pipeline, action.id), null) ?: return emptyMap())
        } catch (_: JSONException) {
            return emptyMap()
        }
        return action.args.associate { spec -> spec.key to spec.coerce(Protocol.optionValueOf(stored.opt(spec.key))) }
    }

    fun save(pipeline: String, action: String, values: Map<String, OptionValue>) {
        prefs.edit().putString(key(pipeline, action), Protocol.valuesJson(values).toString()).apply()
    }

    private fun key(pipeline: String, action: String) = "$pipeline/$action"

    private companion object {
        const val PREFS_NAME = "glass_vision_action_args"
    }
}
