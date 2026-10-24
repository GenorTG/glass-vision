package pl.genor.glassvision

import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/*
 * The in-place translation on the Read text still (Google Lens / Translate style, docs/research/ocr-phone-pipeline.md
 * §5): per block, the background colour sampled around it fills the block, and the translation is drawn inside at the
 * largest size that fits, in a colour that stays legible. The pure math is here (JVM-tested); ReadStillView draws.
 */
object OverlayLayout {
    /** WCAG AA for body text: a sampled glyph colour is kept only with at least this contrast to the background. */
    const val MIN_CONTRAST = 4.5

    const val BLACK = 0xFF000000.toInt()
    const val WHITE = 0xFFFFFFFF.toInt()

    /** Pixels differing from the background by more than this relative luminance count as glyph pixels. */
    const val GLYPH_LUMINANCE = 0.18

    /** At least this share of the inside samples must be glyph pixels for a glyph colour. */
    const val MIN_GLYPH_SHARE = 0.04

    /** WCAG relative luminance 0..1 of an ARGB colour. */
    fun luminance(argb: Int): Double {
        fun channel(c: Int): Double {
            val v = c / 255.0
            return if (v <= 0.03928) v / 12.92 else ((v + 0.055) / 1.055).pow(2.4)
        }
        return 0.2126 * channel(argb shr 16 and 0xFF) + 0.7152 * channel(argb shr 8 and 0xFF) + 0.0722 * channel(argb and 0xFF)
    }

    /** WCAG contrast ratio 1..21. */
    fun contrast(a: Int, b: Int): Double {
        val la = luminance(a)
        val lb = luminance(b)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
    }

    /** The original's glyph colour when it is legible on [background], else black or white, whichever contrasts more. */
    fun textColor(background: Int, glyph: Int?): Int {
        if (glyph != null && contrast(glyph, background) >= MIN_CONTRAST) return glyph or BLACK
        return if (contrast(BLACK, background) >= contrast(WHITE, background)) BLACK else WHITE
    }

    /** Per-channel median of [colors] (opaque), or null if there are none. */
    fun median(colors: List<Int>): Int? {
        if (colors.isEmpty()) return null
        fun mid(shift: Int) = colors.map { it shr shift and 0xFF }.sorted()[colors.size / 2]
        return BLACK or (mid(16) shl 16) or (mid(8) shl 8) or mid(0)
    }

    /** Median of the [inside] samples that stand out from [background] (the glyphs), null if too few do. */
    fun glyphColor(inside: List<Int>, background: Int): Int? {
        val bg = luminance(background)
        val glyphs = inside.filter { kotlin.math.abs(luminance(it) - bg) > GLYPH_LUMINANCE }
        if (glyphs.isEmpty() || glyphs.size < MIN_GLYPH_SHARE * inside.size) return null
        return median(glyphs)
    }

    /** Text laid out at [size]: [lines] in order; [fits] = false when even the minimum size overflows the box. */
    data class Fit(val size: Float, val lines: List<String>, val fits: Boolean)

    /**
     * The largest text size in [minSize]..[maxSize] at which [text] (`\n` = forced line breaks) wraps into
     * [width]×[height] with [lineHeight] (line advance per unit size), never breaking a word (CJK breaks between
     * characters). [measure] = a string's width at size 1. At [minSize] it may still overflow ([Fit.fits] false).
     */
    fun fit(text: String, width: Float, height: Float, measure: (String) -> Float, lineHeight: Float, minSize: Float, maxSize: Float): Fit {
        val paragraphs = text.split('\n').map(::tokens).filter { it.isNotEmpty() }
        if (paragraphs.isEmpty() || width <= 0f || height <= 0f) return Fit(minSize, emptyList(), false)
        val widths = paragraphs.map { p -> p.map { measure(it.text) } }
        val space = measure(" ")
        fun layout(size: Float): List<String>? {
            val out = ArrayList<String>()
            for ((p, tokens) in paragraphs.withIndex()) out += wrap(tokens, widths[p], space, width / size) ?: return null
            return out.takeIf { it.size * lineHeight * size <= height }
        }
        var lo = minSize
        var hi = min(maxSize, height / lineHeight)
        if (hi < lo) hi = lo
        val atMin = layout(lo)
            ?: return Fit(lo, paragraphs.indices.flatMap { p -> wrap(paragraphs[p], widths[p], space, width / lo, force = true).orEmpty() }, false)
        layout(hi)?.let { return Fit(hi, it, true) }
        var best = atMin
        repeat(20) {
            val mid = (lo + hi) / 2f
            val l = layout(mid)
            if (l != null) {
                lo = mid
                best = l
            } else {
                hi = mid
            }
        }
        return Fit(lo, best, true)
    }

    /** A word (or one CJK character) and whether a space separates it from the one before. */
    private data class Token(val text: String, val spaced: Boolean)

    private fun tokens(paragraph: String): List<Token> {
        val out = ArrayList<Token>()
        for (word in paragraph.trim().split(' ').filter { it.isNotEmpty() }) {
            var first = true
            val run = StringBuilder()
            for (c in word) {
                if (isCjk(c)) {
                    if (run.isNotEmpty()) {
                        out += Token(run.toString(), first)
                        first = false
                        run.clear()
                    }
                    out += Token(c.toString(), first && out.isNotEmpty())
                    first = false
                } else {
                    run.append(c)
                }
            }
            if (run.isNotEmpty()) out += Token(run.toString(), first && out.isNotEmpty())
        }
        return out
    }

    /** Greedy wrap at [maxWidth] (units of [widths]); null if a word is wider, unless [force] puts it on its own line. */
    private fun wrap(tokens: List<Token>, widths: List<Float>, space: Float, maxWidth: Float, force: Boolean = false): List<String>? {
        val lines = ArrayList<String>()
        val line = StringBuilder()
        var used = 0f
        for ((i, t) in tokens.withIndex()) {
            val w = widths[i]
            if (w > maxWidth && !force) return null
            val gap = if (line.isEmpty() || !t.spaced) 0f else space
            if (line.isNotEmpty() && used + gap + w > maxWidth) {
                lines += line.toString()
                line.clear()
                used = 0f
            }
            if (line.isNotEmpty() && t.spaced) {
                line.append(' ')
                used += space
            }
            line.append(t.text)
            used += w
        }
        if (line.isNotEmpty()) lines += line.toString()
        return lines
    }

    private fun isCjk(c: Char): Boolean = when (Character.UnicodeScript.of(c.code)) {
        Character.UnicodeScript.HAN, Character.UnicodeScript.HIRAGANA, Character.UnicodeScript.KATAKANA,
        Character.UnicodeScript.HANGUL -> true
        else -> false
    }

    /** Whether ([x], [y]) is inside the polygon [points] (even-odd rule). */
    fun contains(points: List<Pt>, x: Float, y: Float): Boolean {
        var inside = false
        var j = points.size - 1
        for (i in points.indices) {
            val a = points[i]
            val b = points[j]
            if ((a.y > y) != (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x) inside = !inside
            j = i
        }
        return inside
    }

    /** Index of the block whose quad contains the point (the smallest one if they overlap), or -1. */
    fun hit(quads: List<Quad>, x: Float, y: Float): Int =
        quads.indices.filter { contains(quads[it].points, x, y) }.minByOrNull { quads[it].length * quads[it].thickness } ?: -1
}
