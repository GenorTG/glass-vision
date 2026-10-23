package pl.genor.glassvision

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.roundToInt

/** A typed option / action-argument value as it travels on the wire (docs/PROTOCOL.md v3). */
sealed interface OptionValue {
    data class Bool(val value: Boolean) : OptionValue
    data class Num(val value: Double) : OptionValue
    /** `text` and `choice` values. */
    data class Str(val value: String) : OptionValue

    /** Plain value for org.json; whole numbers go out as integers (Python accepts both). */
    fun toJsonValue(): Any = when (this) {
        is Bool -> value
        is Str -> value
        is Num -> if (value == floor(value) && abs(value) < 1e15) value.toLong() else value
    }
}

enum class OptionType(val wire: String) {
    BOOL("bool"), TEXT("text"), NUMBER("number"), CHOICE("choice");

    companion object {
        /** Null for types this app doesn't know: such options are neither shown nor sent. */
        fun fromWire(s: String): OptionType? = entries.firstOrNull { it.wire == s }
    }
}

/** One pipeline option or action argument, described by the server in `hello`. */
data class OptionSpec(
    val key: String,
    val label: String,
    val type: OptionType,
    val default: OptionValue,
    val min: Double? = null,
    val max: Double? = null,
    val step: Double? = null,
    val choices: List<String> = emptyList(),
) {
    /**
     * Returns [value] if the server would accept it for this option, clamped into range for numbers, else
     * [default]. Keeps stale prefs (server changed the option since) from turning into `error` replies.
     */
    fun coerce(value: OptionValue?): OptionValue = when (type) {
        OptionType.BOOL -> value as? OptionValue.Bool ?: default
        OptionType.TEXT -> (value as? OptionValue.Str)?.takeIf { it.value.length <= MAX_TEXT } ?: default
        OptionType.CHOICE -> (value as? OptionValue.Str)?.takeIf { it.value in choices } ?: default
        OptionType.NUMBER -> (value as? OptionValue.Num)?.let { n ->
            var v = n.value
            min?.let { v = v.coerceAtLeast(it) }
            max?.let { v = v.coerceAtMost(it) }
            OptionValue.Num(v)
        } ?: default
    }

    /** A slider needs both bounds; without them a number is edited as text. */
    val isSlider: Boolean get() = type == OptionType.NUMBER && min != null && max != null && max > min

    companion object {
        const val MAX_TEXT = 500
    }
}

/** Integer slider positions for a bounded number option (`min + i * step`). */
class NumberSteps(private val min: Double, private val max: Double, step: Double?) {
    private val step: Double = step?.takeIf { it > 0 } ?: ((max - min) / DEFAULT_STEPS)
    val count: Int = ((max - min) / this.step).roundToInt().coerceAtLeast(1)
    private val decimals: Int = if (this.step >= 1 && this.step == floor(this.step)) 0
    else (-floor(log10(this.step)).toInt()).coerceIn(1, 4)

    fun toProgress(value: Double): Int = ((value - min) / step).roundToInt().coerceIn(0, count)

    fun fromProgress(progress: Int): Double =
        (min + progress.coerceIn(0, count) * step).coerceAtMost(max).let { round(it) }

    fun format(value: Double): String = String.format(java.util.Locale.US, "%.${decimals}f", value)

    private fun round(v: Double): Double = format(v).toDouble()

    companion object {
        private const val DEFAULT_STEPS = 100

        /** "3" for 3.0, "0.25" for 0.25: how a number is pre-filled into a text field. */
        fun plain(v: Double): String =
            if (v == floor(v) && abs(v) < 1e15) v.toLong().toString() else v.toString()
    }
}

data class ActionSpec(val id: String, val label: String, val args: List<OptionSpec> = emptyList())

enum class PipelineStatus(val wire: String) {
    NOT_LOADED("not_loaded"), LOADING("loading"), READY("ready"), ERROR("error"), UNKNOWN("");

    companion object {
        fun fromWire(s: String): PipelineStatus = entries.firstOrNull { it.wire == s && it != UNKNOWN } ?: UNKNOWN
    }
}

/** One recognizer as described by `hello`; the app builds its Recognition UI from this alone. */
data class PipelineSpec(
    val id: String,
    val name: String,
    val description: String = "",
    val defaultEnabled: Boolean = false,
    val usesGlobalConf: Boolean = false,
    val intervalMs: Int = 0,
    val options: List<OptionSpec> = emptyList(),
    val actions: List<ActionSpec> = emptyList(),
    /** Edges between indices of a track's `points`. */
    val skeleton: List<Pair<Int, Int>> = emptyList(),
    val status: PipelineStatus = PipelineStatus.UNKNOWN,
    val statusMessage: String? = null,
)
