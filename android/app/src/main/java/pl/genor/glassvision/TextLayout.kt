package pl.genor.glassvision

import java.util.Locale
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/*
 * The pure half of on-device text reading (OnDeviceText.kt is the Android half): geometry types, crop geometry for
 * Tesseract, which engine reads which OCR language choice, per-line source languages and translation rules.
 * Grouping lines into blocks and reading order: TextBlocks.kt.
 * Plain floats and strings, so all of it is unit-tested on the JVM.
 */

/** A box (left, top, right, bottom): pixels, or normalized 0..1 where documented. */
data class TextBox(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    val width: Float get() = right - left
    val height: Float get() = bottom - top

    fun union(o: TextBox) = TextBox(min(left, o.left), min(top, o.top), max(right, o.right), max(bottom, o.bottom))

    /** This box divided by [w]×[h] (pixels → normalized). */
    fun normalized(w: Float, h: Float) = TextBox(left / w, top / h, right / w, bottom / h)
}

/** A point (pixels or normalized, like the [Quad] it belongs to). */
data class Pt(val x: Float, val y: Float)

/**
 * Four corners of a text region, clockwise from the reading-order top-left (ML Kit `cornerPoints`, PROTOCOL.md
 * `quad`): [points] 0→1 runs along the text, 0→3 down across it. Not necessarily a rectangle.
 */
data class Quad(val points: List<Pt>) {
    init {
        require(points.size == 4) { "a quad has 4 points" }
    }

    fun bounds(): TextBox = TextBox(points.minOf { it.x }, points.minOf { it.y }, points.maxOf { it.x }, points.maxOf { it.y })

    fun map(f: (Pt) -> Pt) = Quad(points.map(f))

    /** Clockwise degrees of the text direction (0 = horizontal, left to right). */
    val angle: Float get() = Math.toDegrees(atan2((points[1].y - points[0].y).toDouble(), (points[1].x - points[0].x).toDouble())).toFloat()

    /** Length along the text (mean of the top and bottom edges). */
    val length: Float get() = (dist(points[0], points[1]) + dist(points[3], points[2])) / 2f

    /** Height across the text (mean of the left and right edges): unlike the box height it doesn't grow with tilt. */
    val thickness: Float get() = (dist(points[0], points[3]) + dist(points[1], points[2])) / 2f

    companion object {
        fun of(b: TextBox) = Quad(listOf(Pt(b.left, b.top), Pt(b.right, b.top), Pt(b.right, b.bottom), Pt(b.left, b.bottom)))

        private fun dist(a: Pt, b: Pt): Float = hypot(a.x - b.x, a.y - b.y)
    }
}

/**
 * One line as an engine read it, in pixels of the image it read; [block] = the engine's block (sign/paragraph),
 * a prior for [TextBlocks]. [quad] = its corners when the engine knows them (ML Kit), [lang] = the engine's language
 * tag (ML Kit `recognizedLanguage`), [script] = the recognizer that read it ("latin", a Tesseract pack code…).
 */
data class OcrLine(
    val text: String,
    val conf: Float,
    val box: TextBox,
    val block: Int,
    val quad: Quad? = null,
    val lang: String? = null,
    val script: String? = null,
) {
    /** [quad], or the box's corners. */
    val corners: Quad get() = quad ?: Quad.of(box)
}

/** Integer crop rectangle [left, right) × [top, bottom) inside an image. */
data class CropRect(val left: Int, val top: Int, val right: Int, val bottom: Int) {
    val width: Int get() = right - left
    val height: Int get() = bottom - top
}

object TextLayout {
    /** Two boxes share a row when they overlap vertically by at least this share of the shorter one. */
    const val ROW_OVERLAP = 0.5f

    /** Crops are padded by this share of the line height on every side (Tesseract wants a margin around glyphs). */
    const val CROP_PAD = 0.25f

    /** Line crops are scaled so the line is about this tall: Tesseract's LSTM reads best around 30–50 px text. */
    const val TARGET_LINE_PX = 48f
    const val MAX_UPSCALE = 4f
    const val MIN_SCALE = 0.5f

    fun sameRow(a: TextBox, b: TextBox): Boolean {
        val overlap = min(a.bottom, b.bottom) - max(a.top, b.top)
        return overlap > 0 && overlap >= ROW_OVERLAP * min(a.height, b.height)
    }

    /** Top to bottom; items sharing a row (see [sameRow]) left to right, or right to left when [rtl]. */
    fun <T> rowOrder(items: List<T>, rtl: Boolean = false, box: (T) -> TextBox): List<T> {
        val rows = ArrayList<MutableList<T>>()
        for (item in items.sortedBy { box(it).top }) {
            val row = rows.lastOrNull()
            if (row != null && row.any { sameRow(box(it), box(item)) }) row += item else rows += mutableListOf(item)
        }
        val across = if (rtl) compareByDescending<T> { box(it).right } else compareBy { box(it).left }
        return rows.flatMap { row -> row.sortedWith(across.thenBy { box(it).top }) }
    }

    /** Share of [a]'s area that [b] covers (0..1). */
    fun covered(a: TextBox, b: TextBox): Float {
        val w = min(a.right, b.right) - max(a.left, b.left)
        val h = min(a.bottom, b.bottom) - max(a.top, b.top)
        val area = a.width * a.height
        return if (w <= 0f || h <= 0f || area <= 0f) 0f else (w * h / area).coerceAtMost(1f)
    }

    /** [line] padded by [CROP_PAD] × its height, clipped to the image; null if nothing usable is left. */
    fun crop(line: TextBox, imageWidth: Int, imageHeight: Int): CropRect? {
        val pad = CROP_PAD * line.height
        val r = CropRect(
            (line.left - pad).toInt().coerceIn(0, imageWidth),
            (line.top - pad).toInt().coerceIn(0, imageHeight),
            Math.ceil((line.right + pad).toDouble()).toInt().coerceIn(0, imageWidth),
            Math.ceil((line.bottom + pad).toDouble()).toInt().coerceIn(0, imageHeight),
        )
        return r.takeIf { it.width >= MIN_CROP_PX && it.height >= MIN_CROP_PX }
    }

    /** Scale for a crop around a line [lineHeight] px tall: small text up (≤ 4×), huge text down (≥ 0.5×). */
    fun cropScale(lineHeight: Float): Float =
        if (lineHeight <= 0f) 1f else (TARGET_LINE_PX / lineHeight).coerceIn(MIN_SCALE, MAX_UPSCALE)

    /** Scaled size of a [width]×[height] crop, at least 1 px each way. */
    fun scaled(width: Int, height: Int, scale: Float): Pair<Int, Int> =
        max(1, (width * scale).roundToInt()) to max(1, (height * scale).roundToInt())

    /** A still is read at up to this long side (like the server). */
    const val READ_LONG_SIDE = 2560

    /** Size to read a [width]×[height] still at: unchanged if it fits [longSide], else scaled down keeping aspect. */
    fun fitLongSide(width: Int, height: Int, longSide: Int = READ_LONG_SIDE): Pair<Int, Int> {
        val long = max(width, height)
        if (long <= longSide) return width to height
        val s = longSide.toFloat() / long
        return max(1, (width * s).roundToInt()) to max(1, (height * s).roundToInt())
    }

    private const val MIN_CROP_PX = 4
}

/** A text recognizer family: ML Kit's (one per script it reads) or Tesseract on line crops ([OTHER]). */
enum class TextScript(val title: String) {
    LATIN("Latin"),
    CHINESE("Chinese"),
    JAPANESE("Japanese"),
    KOREAN("Korean"),
    DEVANAGARI("Devanagari"),
    /** Scripts ML Kit has no recognizer for (Cyrillic, Greek, Arabic, Hebrew, Thai, Bengali, Tamil…). */
    OTHER("Tesseract"),
}

/**
 * How one `ocr_language` choice is read. [script] = the ML Kit recognizer that finds and reads the lines (each
 * non-Latin one also reads Latin). [packs] = Tesseract packs tried on the crops of lines ML Kit read with a
 * confidence under [retryBelow]; a pack's reading replaces ML Kit's only per [TextLanguages.choose]. Packs whose
 * script ML Kit Latin doesn't even find ([OcrLanguage.latinFindsLines]) also scan the whole image. [fallback] =
 * packs used like [packs] (on every line) while [script]'s recognizer is unavailable (no Google Play services).
 * [auto]: a pack that can't be loaded is skipped instead of failing the read.
 */
data class OcrPlan(
    val script: TextScript,
    val packs: List<OcrLanguage> = emptyList(),
    val retryBelow: Float = RETRY_BELOW,
    val fallback: List<OcrLanguage> = emptyList(),
    val auto: Boolean = false,
) {
    companion object {
        /**
         * Lines ML Kit read with at least this confidence keep its reading: on rendered signs Latin lines scored
         * 0.80–0.84 and its misreadings of Cyrillic/Greek/Tamil 0.32–0.55 (camera photos score lower overall).
         */
        const val RETRY_BELOW = 0.7f

        /** Auto mode tries at most this many packs (each costs a crop read per unsure line, some a full scan). */
        const val AUTO_MAX_PACKS = 3

        /**
         * - A Latin-script language: ML Kit Latin only (it reads all of them; no pack download).
         * - Chinese / Japanese / Korean / Hindi: that ML Kit recognizer (Play services); the language's pack only as
         *   [fallback].
         * - Any other script (Russian, Arabic, Greek…): ML Kit Latin finds the lines, the chosen pack reads the
         *   ones ML Kit isn't sure of; ML Kit's reading stays for Latin words on a Cyrillic sign.
         * - Auto-detect: the phone's first language picks the ML Kit recognizer (a Japanese phone → Japanese, else
         *   Latin); lines read with low confidence are retried on the packs of the phone's languages, then on packs
         *   already downloaded ([installed], most recently used first), at most [AUTO_MAX_PACKS] — no download of
         *   any other pack (like the server's auto mode).
         */
        fun forChoice(choice: String, installed: List<OcrLanguage>, phoneLanguages: List<String>): OcrPlan {
            if (choice != OcrLanguage.AUTO_CHOICE) {
                val lang = OcrLanguage.fromChoice(choice)
                return when (lang.script) {
                    TextScript.LATIN -> OcrPlan(TextScript.LATIN)
                    TextScript.OTHER -> OcrPlan(TextScript.LATIN, packs = listOf(lang))
                    else -> OcrPlan(lang.script, fallback = listOf(lang))
                }
            }
            val phone = phoneLanguages.mapNotNull(OcrLanguage::fromLanguageTag)
            val first = phone.firstOrNull()?.takeIf { it.script != TextScript.OTHER && it.script != TextScript.LATIN }
            val packs = (phone + installed).filter { it.script == TextScript.OTHER }.distinct().take(AUTO_MAX_PACKS)
            return OcrPlan(
                script = first?.script ?: TextScript.LATIN,
                packs = packs,
                fallback = listOfNotNull(first),
                auto = true,
            )
        }
    }
}

/** A line reading with its confidence 0..1 (ML Kit's, or Tesseract's mean word confidence / 100). */
data class Reading(val text: String, val conf: Float)

object TextLanguages {
    /**
     * A pack reading needs at least this confidence (Tesseract read the right script on rendered signs at
     * 0.81–0.96, wrong-script crops at ≤ 0.72).
     */
    const val MIN_PACK_CONF = 0.6f

    /**
     * ML Kit's reading of a line or one of the Tesseract [packs] readings of its crop: the most confident pack
     * reading that is mostly in its pack's script ([inScript]), at least [MIN_PACK_CONF] and more confident than
     * ML Kit; else ML Kit's. (A Cyrillic pack reads "EXIT" as "ЕХІТ" at 0.94: confidence alone can't decide.)
     */
    fun choose(mlKit: Reading, packs: List<Pair<OcrLanguage, Reading>>): Reading =
        packs.filter { (lang, r) -> r.conf >= MIN_PACK_CONF && r.conf > mlKit.conf && inScript(r.text, lang) }
            .maxByOrNull { it.second.conf }?.second ?: mlKit

    /** Whether at least half of [text]'s letters are in [lang]'s scripts (false without letters). */
    fun inScript(text: String, lang: OcrLanguage): Boolean {
        var letters = 0
        var hits = 0
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            i += Character.charCount(cp)
            if (!Character.isLetter(cp)) continue
            letters++
            if (Character.UnicodeScript.of(cp) in lang.letters) hits++
        }
        return letters > 0 && hits * 2 >= letters
    }

    /** An ML Kit language-ID tag → a source code to translate from, or null ("und", romanized "ja-Latn"…). */
    fun sourceCode(tag: String?): String? {
        if (tag.isNullOrBlank() || tag == UNDETERMINED) return null
        val parts = tag.split('-')
        // "ru-Latn", "ja-Latn"…: romanized text, which no translation model takes.
        if (parts.drop(1).any { it.equals("Latn", ignoreCase = true) }) return null
        return parts[0].lowercase(Locale.ROOT)
    }

    /** Letters in [text]. */
    fun letters(text: String): Int = text.count(Char::isLetter)

    /** Identification on fewer letters than this is unreliable: such lines inherit their block's language. */
    const val MIN_ID_LETTERS = 8

    /**
     * Source language per line ([lines] in reading order, their blocks set): long-enough lines are identified on
     * their own; the rest inherit their block's language (identified on the whole block's text), then the
     * language most lines of the still have — one-word lines ("Ausgang") are unreliable alone. [identify] maps
     * text to a code or null; where it has no answer for a long line, the engine's own tag ([OcrLine.lang]) counts.
     */
    fun assign(lines: List<OcrLine>, identify: (String) -> String?): List<String?> {
        val own = lines.map { if (letters(it.text) >= MIN_ID_LETTERS) identify(it.text) ?: sourceCode(it.lang) else null }
        val blockLang = HashMap<Int, String?>()
        for ((block, members) in lines.indices.groupBy { lines[it].block }) {
            val known = members.mapNotNull { own[it] }
            blockLang[block] = majority(known) ?: run {
                val text = members.joinToString(" ") { lines[it].text }
                if (letters(text) >= MIN_ID_LETTERS) identify(text) else null
            }
        }
        val still = majority(lines.indices.mapNotNull { own[it] ?: blockLang[lines[it].block] })
        return lines.indices.map { i ->
            own[i] ?: blockLang[lines[i].block] ?: if (letters(lines[i].text) > 0) still else null
        }
    }

    private fun majority(codes: List<String>): String? =
        codes.groupingBy { it }.eachCount().maxWithOrNull(compareBy<Map.Entry<String, Int>> { it.value })?.key

    /**
     * Whether a line is worth translating from [source] into [target]: it has letters, more letters than digits
     * ("A4 2024" stays), a known source and a different language.
     */
    fun translatable(text: String, source: String?, target: String): Boolean {
        val letters = letters(text)
        if (letters < 2 || text.count(Char::isDigit) > letters) return false
        return source != null && source != target
    }

    /**
     * Translation targets for language [codes] (ML Kit's TranslateLanguage tags): named in English like the
     * server's list, sorted by name.
     */
    fun targets(codes: Collection<String>): List<TranslationLanguage> =
        codes.distinct().map { TranslationLanguage(it, name(it)) }.sortedBy { it.name }

    fun name(code: String): String =
        Locale.forLanguageTag(code).getDisplayLanguage(Locale.ENGLISH).takeIf { it.isNotBlank() && it != code } ?: code

    const val UNDETERMINED = "und"
}
