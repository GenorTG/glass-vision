package pl.genor.glassvision

import android.graphics.Bitmap

/**
 * One recognized line of a still (docs/PROTOCOL.md "Read text"). [box]/[quad] are normalized 0..1 against the still
 * as captured (upright, before any rotation the reader applied itself).
 */
data class ReadLine(
    val text: String,
    val conf: Float,
    val box: TextBox,
    /** Lines with the same block belong together (one sign / paragraph). */
    val block: Int,
    /** Detected source language code (ISO 639-1, e.g. "de"), or null if unknown. */
    val lang: String?,
    /** The line's corners (v3.3), null when the reader only gave a box. */
    val quad: Quad? = null,
    /** Recognizer that read the line ("latin", "cyrillic", a Tesseract pack…), null if unknown. */
    val script: String? = null,
    /** The reading looks like gibberish ([TextSuspect], the server's detector). */
    val suspect: Boolean = false,
)

/** How a block is translated and shown (PROTOCOL.md v3.3 `kind`). */
enum class BlockKind(val wire: String) {
    /** A sentence or paragraph: translated as one text, shown reflowed. */
    PROSE("prose"),

    /** Menu, timetable, stacked sign words: translated and shown line by line. */
    LIST("list");

    companion object {
        fun of(wire: String?): BlockKind? = entries.firstOrNull { it.wire == wire }
    }
}

/**
 * One sign/paragraph — what the user sees as one object. [lines] = indices into [TextReading.lines], in reading
 * order; [box]/[quad] normalized like the lines; [text] = the lines joined with `\n` (as read, hyphens kept).
 * [translations]: target code → this block's translation, or null where it isn't translated (already in that
 * language, digits only…); a code is present for every target that succeeded.
 */
data class ReadBlock(
    val id: Int,
    val lines: List<Int>,
    val box: TextBox = TextBox(0f, 0f, 0f, 0f),
    val quad: Quad? = null,
    val text: String = "",
    val kind: BlockKind = BlockKind.LIST,
    val lang: String? = null,
    val suspect: Boolean = false,
    val translations: Map<String, String?> = emptyMap(),
) {
    /** [quad], or the corners of [box]. */
    val corners: Quad get() = quad ?: Quad.of(box)

    /** What a reader of this block in [kind] should get: prose joined into one paragraph, a list line by line. */
    val display: String get() = if (kind == BlockKind.PROSE) TextBlocks.joinProse(text.split('\n')) else text
}

/**
 * Result of reading one still: lines and the [blocks] they form (always set: readers without blocks get one block per
 * `block` id, [ReadBlocks.fromLines]). Translations live in the blocks.
 */
data class TextReading(
    val width: Int,
    val height: Int,
    val lines: List<ReadLine>,
    val blocks: List<ReadBlock>,
    /** Target code → why that whole translation failed. */
    val errors: Map<String, String> = emptyMap(),
    /** Script/recognizer that read it (server: "latin", "cyrillic"…; phone: engine name), for display only. */
    val script: String? = null,
    /** Clockwise degrees the reader rotated the still before reading (0/90/180/270 + fine skew), for display only. */
    val angle: Float = 0f,
) {
    /** Targets that were translated (every block has an entry for them, possibly null). */
    val translated: Set<String> get() = blocks.flatMapTo(LinkedHashSet()) { it.translations.keys }

    /** [code]'s translation per block (null = not translated), aligned with [blocks]. */
    fun translation(code: String): List<String?> = blocks.map { it.translations[code] }
}

/** A translation target the reader offers. */
data class TranslationLanguage(val code: String, val name: String)

/**
 * Reads text on a still and translates it — the backend half of the Read button. Every method blocks (HTTP or
 * on-device inference): call from a worker thread, never the main thread. Failures throw; the UI shows the message.
 */
interface TextReader {
    /** OCR [still] (upright) and translate its blocks into each of [translateTo] (codes from [languages]). */
    fun read(still: Bitmap, translateTo: List<String>): TextReading

    /** Translate already-read [lines] into [to]; [from] = per-line source codes (null entries = detect). */
    fun translate(lines: List<String>, from: List<String?>, to: String): List<String?>

    /**
     * Translate already-read [blocks] into [to] ("Add language"), one entry per block, null = not translated. This
     * default goes line by line through [translate] (readers without block translation); readers that can, translate
     * prose blocks as one text.
     */
    fun translateBlocks(blocks: List<ReadBlock>, to: String): List<String?> {
        val split = blocks.map { it.text.split('\n') }
        val flat = translate(split.flatten(), split.flatMapIndexed { i, l -> List(l.size) { blocks[i].lang } }, to)
        var next = 0
        return split.map { lines ->
            val out = lines.map { line -> line to flat.getOrNull(next++) }
            if (out.all { it.second == null }) null else out.joinToString("\n") { it.second ?: it.first }
        }
    }

    /** Targets [read]/[translate] accept, sorted by name. */
    fun languages(): List<TranslationLanguage>
}
