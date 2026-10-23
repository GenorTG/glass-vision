package pl.genor.glassvision

import kotlin.math.roundToInt

/**
 * How a track looks, independent of Canvas: colour = a base hue per pipeline (golden-angle steps over the
 * order in `hello`, so the first few pipelines are far apart) nudged by the label, so a person and a cup
 * from "objects" differ slightly while staying in the objects family. JVM-testable.
 */
object TrackStyle {
    /** OCR text: long labels, thin boxes. */
    const val TEXT_PIPELINE = "text"
    private const val GOLDEN_ANGLE = 137.508f
    private const val LABEL_SPREAD_STEPS = 7 // label offsets -3..+3
    private const val LABEL_STEP_DEG = 8f

    /** Hue 0..360 for [pipelineIndex] (position in hello, or -1 if unknown) and [label]. */
    fun hue(pipeline: String, pipelineIndex: Int, label: String): Float {
        val slot = if (pipelineIndex >= 0) pipelineIndex else positiveHash(pipeline) % 16
        val base = slot * GOLDEN_ANGLE
        val offset = (positiveHash(label) % LABEL_SPREAD_STEPS - LABEL_SPREAD_STEPS / 2) * LABEL_STEP_DEG
        return ((base + offset) % 360f + 360f) % 360f
    }

    /** "label · detail 91%"; the detail only when present and different from the label. */
    fun labelText(d: Detection): String = buildString {
        append(d.label.ifEmpty { d.pipeline.ifEmpty { "?" } })
        d.detail?.takeIf { it.isNotBlank() && it != d.label }?.let { append(" · ").append(it) }
        append(' ').append((d.conf * 100).roundToInt()).append('%')
    }

    fun isThin(d: Detection): Boolean = d.pipeline == TEXT_PIPELINE

    /** A "Mark text areas" box (label = the server's/phone's placeholder "text") is just the box: no chip. */
    fun hasLabel(d: Detection): Boolean = !(d.pipeline == TEXT_PIPELINE && d.label == MARK_LABEL)

    private const val MARK_LABEL = "text"

    /** String.hashCode is specified (stable across runs/devices), unlike identity hashes. */
    private fun positiveHash(s: String): Int = s.hashCode() and 0x7FFF_FFFF
}
