package pl.genor.glassvision

import android.graphics.Bitmap
import android.graphics.RectF

/** One recognized line of a still (docs/PROTOCOL.md "Read text"). [box] is normalized 0..1 against the still. */
data class ReadLine(
    val text: String,
    val conf: Float,
    val box: RectF,
    /** Lines with the same block belong together (one sign / paragraph). */
    val block: Int,
    /** Detected source language code (ISO 639-1, e.g. "de"), or null if unknown. */
    val lang: String?,
)

/** Result of reading one still. [translations] maps a target code to entries aligned with [lines] (null = untranslated). */
data class TextReading(
    val width: Int,
    val height: Int,
    val lines: List<ReadLine>,
    val translations: Map<String, List<String?>>,
    /** Target code → why that whole translation failed. */
    val errors: Map<String, String> = emptyMap(),
    /** Script/recognizer that read it (server: "latin", "cyrillic"…; phone: engine name), for display only. */
    val script: String? = null,
)

/** A translation target the reader offers. */
data class TranslationLanguage(val code: String, val name: String)

/**
 * Reads text on a still and translates it — the backend half of the Read button. Every method blocks (HTTP or
 * on-device inference): call from a worker thread, never the main thread. Failures throw; the UI shows the message.
 */
interface TextReader {
    /** OCR [still] (upright) and translate the lines into each of [translateTo] (codes from [languages]). */
    fun read(still: Bitmap, translateTo: List<String>): TextReading

    /** Translate already-read [lines] into [to]; [from] = per-line source codes (null entries = detect). */
    fun translate(lines: List<String>, from: List<String?>, to: String): List<String?>

    /** Targets [read]/[translate] accept, sorted by name. */
    fun languages(): List<TranslationLanguage>
}
