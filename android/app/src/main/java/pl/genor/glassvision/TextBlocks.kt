package pl.genor.glassvision

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/*
 * Lines → blocks on the phone (docs/research/ocr-phone-pipeline.md §4): ML Kit often makes every line of a sign its
 * own block and splits one row at a wide gap, so blocks are rebuilt from the lines' quads. Pure Kotlin, JVM-tested.
 */

/** One block found on the phone: [lines] = indices into [OcrGrouping.lines] (contiguous, in order), [quad] in pixels. */
data class OcrBlock(val lines: List<Int>, val quad: Quad, val kind: BlockKind)

/** Lines in reading order ([OcrLine.block] = index into [blocks]), the blocks, and the skew [angle] (clockwise °). */
data class OcrGrouping(val lines: List<OcrLine>, val blocks: List<OcrBlock>, val angle: Float) {
    /** A block's lines joined with `\n` (PROTOCOL.md v3.3 block `text`). */
    fun text(block: OcrBlock): String = block.lines.joinToString("\n") { lines[it].text }
}

object TextBlocks {
    /** Row pieces: baselines closer than this share of the line height… */
    const val ROW_BASELINE = 0.3f
    /** …heights within this factor… */
    const val ROW_HEIGHT_RATIO = 1.3f
    /** …and a horizontal gap under this many line heights (wider = a column gutter) are one line. */
    const val ROW_GAP = 1.2f

    /**
     * Stacked lines join a block across a gap up to this many line heights (of the smaller text)… (the research note
     * suggests 0.8; signs set at 1.4× leading measure 0.85–0.9 line heights between the lines' boxes)
     */
    const val BLOCK_GAP = 1.0f
    /** …or this many when the engine put them in one block already. */
    const val PRIOR_GAP = 1.5f
    /** Heights differing by more than this factor are a heading and its body, or different signs. */
    const val BLOCK_HEIGHT_RATIO = 1.35f
    /** Left edges, centres or right edges within this many line heights are aligned… */
    const val ALIGN = 1.0f
    /** …as is a horizontal overlap of at least this share of the narrower line. */
    const val MIN_OVERLAP = 0.3f

    /** Prose: every line but the last fills at least this share of the block width. */
    const val PROSE_FILL = 0.7f

    /**
     * Groups [input] lines (pixels of one image, engine order): deskew by the median line angle → stitch pieces of
     * one row → link stacked lines into blocks → recursive XY-cut reading order (right to left for RTL scripts) →
     * prose/list per block. Blank lines are dropped.
     */
    fun group(input: List<OcrLine>): OcrGrouping {
        val lines = input.filter { it.text.isNotBlank() }
        if (lines.isEmpty()) return OcrGrouping(emptyList(), emptyList(), 0f)
        val angle = skew(lines)
        val frame = Frame(angle)
        val rows = stitch(lines.map { Geo.of(it, frame) }, frame)
        val hulls = link(rows).map { members -> Hull(members, members.map { rows[it].box }.reduce(TextBox::union)) }
        val outLines = ArrayList<OcrLine>(rows.size)
        val blocks = ArrayList<OcrBlock>(hulls.size)
        for ((b, hull) in xyCut(hulls, isRtl(lines.joinToString(" ") { it.text })).withIndex()) {
            val rtl = isRtl(hull.members.joinToString(" ") { rows[it].line.text })
            val ordered = TextLayout.rowOrder(hull.members, rtl) { rows[it].box }
            val first = outLines.size
            ordered.forEach { outLines += rows[it].line.copy(block = b) }
            val kind = kind(ordered.map { rows[it].line.text }, ordered.map { rows[it].box.width }, hull.box.width)
            blocks += OcrBlock((first until outLines.size).toList(), frame.back(hull.box), kind)
        }
        return OcrGrouping(outLines, blocks, angle)
    }

    /**
     * Prose or list ([BlockKind]): one line = prose; a line ending in `:` or a digit (menus, timetables) = list; a
     * hyphen/comma at a line end or a lowercase line start (a wrapped sentence) = prose; otherwise prose only if every
     * line but the last fills [PROSE_FILL] of the block and lines average 2+ words (stacked sign words are a list).
     */
    fun kind(texts: List<String>, widths: List<Float>, blockWidth: Float): BlockKind {
        if (texts.size < 2) return BlockKind.PROSE
        val trimmed = texts.map(String::trim)
        if (trimmed.any { it.endsWith(':') || it.lastOrNull()?.isDigit() == true }) return BlockKind.LIST
        val last = trimmed.size - 1
        val continued = (0 until last).any { i ->
            trimmed[i].endsWith('-') || trimmed[i].endsWith(',') || trimmed[i + 1].firstOrNull()?.isLowerCase() == true
        }
        if (continued) return BlockKind.PROSE
        if ((0 until last).any { widths[it] < PROSE_FILL * blockWidth }) return BlockKind.LIST
        return if (trimmed.sumOf(::words).toFloat() / trimmed.size < 2f) BlockKind.LIST else BlockKind.PROSE
    }

    /**
     * A prose block as one text: lines joined with spaces, a line-end hyphen before a lowercase word removed
     * ("Bahn- / steig" → "Bahnsteig"; kept before a capital or a conjunction: "Nord- / Süd", "Ein- / und Ausgang"),
     * no space between CJK characters.
     */
    fun joinProse(lines: List<String>): String = buildString {
        for (raw in lines) {
            val line = raw.trim()
            if (line.isEmpty()) continue
            if (isEmpty()) {
                append(line)
                continue
            }
            val prev = last()
            when {
                (prev == '-' || prev == '\u00AD') && length >= 2 && this[length - 2].isLetter() && line.first().isLowerCase() &&
                    line.substringBefore(' ').lowercase() !in CONJUNCTIONS -> {
                    deleteCharAt(length - 1)
                    append(line)
                }
                isCjk(prev) && isCjk(line.first()) -> append(line)
                else -> append(' ').append(line)
            }
        }
    }

    /** Whether most strongly directional letters of [text] are right to left (Arabic, Hebrew…). */
    fun isRtl(text: String): Boolean {
        var rtl = 0
        var ltr = 0
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            i += Character.charCount(cp)
            when (Character.getDirectionality(cp)) {
                Character.DIRECTIONALITY_RIGHT_TO_LEFT, Character.DIRECTIONALITY_RIGHT_TO_LEFT_ARABIC -> rtl++
                Character.DIRECTIONALITY_LEFT_TO_RIGHT -> ltr++
            }
        }
        return rtl > ltr
    }

    /**
     * The still's skew: the weighted median of the lines' angles (weight = confidence × length; ML Kit's confidence
     * 0 = unknown counts as 0.3), around their circular mean so ±180° don't average to 0. Lines without a quad don't
     * vote; none → 0.
     */
    fun skew(lines: List<OcrLine>): Float {
        val voters = lines.mapNotNull { l -> l.quad?.let { q -> q.angle to max(l.conf, 0.3f) * max(q.length, 1f) } }
        if (voters.isEmpty()) return 0f
        var sx = 0.0
        var sy = 0.0
        for ((a, w) in voters) {
            sx += w * cos(Math.toRadians(a.toDouble()))
            sy += w * sin(Math.toRadians(a.toDouble()))
        }
        val mean = Math.toDegrees(kotlin.math.atan2(sy, sx)).toFloat()
        val deltas = voters.map { (a, w) -> wrap(a - mean) to w }.sortedBy { it.first }
        val half = deltas.sumOf { it.second.toDouble() } / 2
        var acc = 0.0
        val median = deltas.first { acc += it.second; acc >= half }.first
        return wrap(mean + median)
    }

    private fun wrap(deg: Float): Float {
        var d = deg % 360f
        if (d > 180f) d -= 360f
        if (d <= -180f) d += 360f
        return d
    }

    /** Words after a suspended hyphen ("Ein- und Ausgang", "pre- or post-war"): the hyphen stays. */
    private val CONJUNCTIONS = setOf("und", "oder", "bis", "sowie", "and", "or", "to", "et", "ou", "i", "lub", "oraz", "a", "e", "o", "y", "en", "of")

    private fun words(text: String): Int = max(text.split(' ').count { it.any(Char::isLetterOrDigit) }, text.count(::isCjk) / 2)

    private fun isCjk(c: Char): Boolean = when (Character.UnicodeScript.of(c.code)) {
        Character.UnicodeScript.HAN, Character.UnicodeScript.HIRAGANA, Character.UnicodeScript.KATAKANA -> true
        else -> false
    }

    /** Rotation by −[angle] (into the deskewed frame) and back. */
    private class Frame(angle: Float) {
        private val c = cos(Math.toRadians(angle.toDouble())).toFloat()
        private val s = sin(Math.toRadians(angle.toDouble())).toFloat()

        fun to(p: Pt) = Pt(p.x * c + p.y * s, -p.x * s + p.y * c)

        fun from(p: Pt) = Pt(p.x * c - p.y * s, p.x * s + p.y * c)

        /** A deskewed box as a quad in image pixels. */
        fun back(b: TextBox): Quad = Quad.of(b).map(::from)
    }

    /** A line in the deskewed frame: [box] = its bounds there, [h] = its thickness, [baseline] = bottom edge y. */
    private class Geo(val line: OcrLine, val box: TextBox, val h: Float, val baseline: Float) {
        val centerX: Float get() = (box.left + box.right) / 2f
        val centerY: Float get() = (box.top + box.bottom) / 2f

        companion object {
            fun of(line: OcrLine, frame: Frame, h: Float? = null): Geo {
                val q = line.corners.map(frame::to)
                return Geo(line, q.bounds(), h ?: line.corners.thickness, (q.points[2].y + q.points[3].y) / 2f)
            }
        }
    }

    private class Hull(val members: List<Int>, val box: TextBox)

    /** Step 2: pieces of one row (left to right, the nearest fitting row end first) merged into one line. */
    private fun stitch(geos: List<Geo>, frame: Frame): List<Geo> {
        val rows = ArrayList<MutableList<Geo>>()
        for (g in geos.sortedBy { it.box.left }) {
            var best = -1
            var bestGap = Float.MAX_VALUE
            for ((i, row) in rows.withIndex()) {
                val end = row.last()
                val h = min(end.h, g.h)
                if (h <= 0f || max(end.h, g.h) / h >= ROW_HEIGHT_RATIO) continue
                if (abs(end.baseline - g.baseline) >= ROW_BASELINE * h) continue
                val gap = g.box.left - end.box.right
                if (gap >= ROW_GAP * h || gap < -0.5f * h) continue
                if (gap < bestGap) {
                    best = i
                    bestGap = gap
                }
            }
            if (best >= 0) rows[best] += g else rows += mutableListOf(g)
        }
        return rows.map { merge(it, frame) }
    }

    private fun merge(parts: List<Geo>, frame: Frame): Geo {
        if (parts.size == 1) return parts[0]
        val ordered = if (isRtl(parts.joinToString(" ") { it.line.text })) parts.reversed() else parts
        val known = parts.filter { it.line.conf > 0f }
        val conf = if (known.isEmpty()) 0f else known.sumOf { (it.line.conf * it.box.width).toDouble() }.toFloat() / known.sumOf { it.box.width.toDouble() }.toFloat()
        val box = parts.map { it.box }.reduce(TextBox::union)
        val quad = frame.back(box)
        val main = parts.maxBy { TextLanguages.letters(it.line.text) }.line
        val line = OcrLine(
            text = ordered.joinToString(" ") { it.line.text.trim() },
            conf = conf,
            box = quad.bounds(),
            block = main.block,
            quad = quad,
            lang = parts.mapNotNull { it.line.lang }.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key,
            script = main.script,
        )
        return Geo.of(line, frame, h = parts.sumOf { it.h.toDouble() }.toFloat() / parts.size)
    }

    /** Step 3: union-find over stacked, aligned lines of a similar size with nothing in between. */
    private fun link(rows: List<Geo>): List<List<Int>> {
        val parent = IntArray(rows.size) { it }
        fun root(i: Int): Int {
            var r = i
            while (parent[r] != r) r = parent[r]
            return r
        }
        for (i in rows.indices) {
            for (j in i + 1 until rows.size) {
                val (a, b) = if (rows[i].centerY <= rows[j].centerY) i to j else j to i
                if (linked(a, b, rows)) parent[root(a)] = root(b)
            }
        }
        return rows.indices.groupBy(::root).values.toList()
    }

    private fun linked(ia: Int, ib: Int, rows: List<Geo>): Boolean {
        val a = rows[ia]
        val b = rows[ib]
        val h = min(a.h, b.h)
        if (h <= 0f || max(a.h, b.h) / h > BLOCK_HEIGHT_RATIO) return false
        val gap = b.box.top - a.box.bottom
        val maxGap = (if (a.line.block == b.line.block) PRIOR_GAP else BLOCK_GAP) * h
        if (gap > maxGap || gap < -0.5f * h) return false
        val overlap = min(a.box.right, b.box.right) - max(a.box.left, b.box.left)
        val aligned = abs(a.box.left - b.box.left) <= ALIGN * h ||
            abs(a.centerX - b.centerX) <= ALIGN * h ||
            abs(a.box.right - b.box.right) <= ALIGN * h ||
            overlap >= MIN_OVERLAP * min(a.box.width, b.box.width)
        if (!aligned) return false
        // Something of a comparable size between them (another sign's line, a heading): not one block.
        val left = min(a.box.left, b.box.left)
        val right = max(a.box.right, b.box.right)
        return rows.indices.none { k ->
            val c = rows[k]
            k != ia && k != ib && c.centerY > a.centerY && c.centerY < b.centerY && c.h >= 0.5f * h &&
                c.box.top >= a.box.bottom - 0.5f * h && c.box.bottom <= b.box.top + 0.5f * h &&
                min(c.box.right, right) - max(c.box.left, left) > 0f
        }
    }

    /**
     * Step 4: recursive XY-cut — split at the widest whitespace gap across all blocks (rows top to bottom, columns
     * left to right or right to left when [rtl]; a horizontal gap wins a tie), recurse; without any gap, rows.
     */
    private fun xyCut(items: List<Hull>, rtl: Boolean): List<Hull> {
        if (items.size <= 1) return items
        val y = widestGap(items.map { it.box.top to it.box.bottom })
        val x = widestGap(items.map { it.box.left to it.box.right })
        if (y == null && x == null) return TextLayout.rowOrder(items, rtl) { it.box }
        return if (y != null && (x == null || y.second >= x.second)) {
            val (above, below) = items.partition { it.box.top < y.first }
            xyCut(above, rtl) + xyCut(below, rtl)
        } else {
            val (left, right) = items.partition { it.box.left < x!!.first }
            if (rtl) xyCut(right, rtl) + xyCut(left, rtl) else xyCut(left, rtl) + xyCut(right, rtl)
        }
    }

    /** The widest gap between the union of [intervals]: (cut position, gap size), or null if they all overlap. */
    private fun widestGap(intervals: List<Pair<Float, Float>>): Pair<Float, Float>? {
        val sorted = intervals.sortedBy { it.first }
        var end = sorted.first().second
        var best: Pair<Float, Float>? = null
        for ((start, stop) in sorted.drop(1)) {
            val gap = start - end
            if (gap > 0f && (best == null || gap > best.second)) best = (end + start) / 2f to gap
            end = max(end, stop)
        }
        return best
    }
}

/** A phone read in the [TextReading] shape: normalized lines and blocks (no translations yet). */
data class LocalReading(val lines: List<ReadLine>, val blocks: List<ReadBlock>) {
    companion object {
        /**
         * [grouping] of an image [w]×[h] px with per-line source languages [sources] → normalized lines (with quads,
         * [TextSuspect] verdicts) and blocks (language = their lines' majority).
         */
        fun of(grouping: OcrGrouping, sources: List<String?>, w: Float, h: Float): LocalReading {
            fun norm(p: Pt) = Pt((p.x / w).coerceIn(0f, 1f), (p.y / h).coerceIn(0f, 1f))
            val lines = grouping.lines.mapIndexed { i, l ->
                ReadLine(
                    text = l.text,
                    conf = l.conf,
                    box = l.corners.map(::norm).bounds(),
                    block = l.block,
                    lang = sources.getOrNull(i),
                    quad = l.quad?.map(::norm),
                    script = l.script,
                    suspect = TextSuspect.line(l.text, l.conf),
                )
            }
            val blocks = grouping.blocks.mapIndexed { b, block ->
                val quad = block.quad.map(::norm)
                ReadBlock(
                    id = b,
                    lines = block.lines,
                    box = quad.bounds(),
                    quad = quad,
                    text = grouping.text(block),
                    kind = block.kind,
                    lang = ReadBlocks.majority(block.lines.mapNotNull { sources.getOrNull(it) }),
                    suspect = TextSuspect.block(block.lines.map { lines[it].text to lines[it].suspect }),
                )
            }
            return LocalReading(lines, blocks)
        }
    }
}

/**
 * The phone's gibberish check (PROTOCOL.md v3.3 `suspect`, same idea as the server's): a line is suspect when its
 * reading looks unlike real text — low recognizer confidence, letters of two scripts inside one word ("ЕХIТ"), case
 * flips inside a word ("BbIXOI" = ВЫХОД read as Latin), digits inside lowercase words ("c3ro"), mostly symbols, a long
 * Latin word without vowels, a character repeated 4+ times. Each sign adds points; [SUSPECT_SCORE] points = suspect.
 */
object TextSuspect {
    const val SUSPECT_SCORE = 2

    /**
     * Below this (known, > 0) confidence a reading is unsure; below [VERY_LOW_CONF] very unsure. ML Kit Latin read
     * rendered Latin signs at 0.80–0.84 and misread Cyrillic/Greek/Tamil ones at 0.32–0.55 (OcrPlan.RETRY_BELOW).
     */
    const val LOW_CONF = 0.6f
    const val VERY_LOW_CONF = 0.35f

    private val WORD_SPLIT = Regex("\\s+")
    private val DIGIT_INSIDE = Regex("\\p{Ll}\\d+\\p{Ll}")
    private val REPEATED = Regex("(\\S)\\1{3,}")
    private const val VOWELS = "aeiouyàáâãäåæèéêëěìíîïòóôõöøœùúûüůýÿąęėįųіı"

    fun line(text: String, conf: Float): Boolean = score(text, conf) >= SUSPECT_SCORE

    /** A block is suspect when its suspect lines hold at least half of its letters (or, letterless, half its lines). */
    fun block(lines: List<Pair<String, Boolean>>): Boolean {
        if (lines.isEmpty()) return false
        val letters = lines.sumOf { TextLanguages.letters(it.first) }
        if (letters == 0) return lines.count { it.second } * 2 >= lines.size
        return lines.filter { it.second }.sumOf { TextLanguages.letters(it.first) } * 2 >= letters
    }

    fun score(text: String, conf: Float): Int {
        var s = 0
        if (conf > 0f && conf < LOW_CONF) s += if (conf < VERY_LOW_CONF) 2 else 1
        val words = text.trim().split(WORD_SPLIT).filter { it.isNotEmpty() }
        if (words.any(::mixedScripts)) s += 2
        if (words.any(::caseFlips)) s += 1
        if (DIGIT_INSIDE.containsMatchIn(text)) s += 1
        if (words.any(::noVowels)) s += 1
        if (REPEATED.containsMatchIn(text)) s += 1
        val visible = text.count { !it.isWhitespace() }
        if (visible >= 4 && text.count { !it.isWhitespace() && !it.isLetterOrDigit() } * 10 > visible * 3) s += 1
        return s
    }

    /** Letters of two scripts in one word (Han, kana and Hangul count as one: Japanese and Korean mix them). */
    private fun mixedScripts(word: String): Boolean {
        val scripts = HashSet<Any>()
        var i = 0
        while (i < word.length) {
            val cp = word.codePointAt(i)
            i += Character.charCount(cp)
            if (!Character.isLetter(cp)) continue
            scripts += when (val sc = Character.UnicodeScript.of(cp)) {
                Character.UnicodeScript.HAN, Character.UnicodeScript.HIRAGANA, Character.UnicodeScript.KATAKANA,
                Character.UnicodeScript.HANGUL -> "cjk"
                else -> sc
            }
        }
        return scripts.size >= 2
    }

    /** "bIXOI": a lowercase letter followed by 2+ capitals, or two lowercase→capital flips ("iPhone", "McDonald" pass). */
    private fun caseFlips(word: String): Boolean {
        val letters = word.filter(Char::isLetter)
        var flips = 0
        for (i in 1 until letters.length) {
            if (letters[i - 1].isLowerCase() && letters[i].isUpperCase()) {
                flips++
                if (i + 1 < letters.length && letters[i + 1].isUpperCase()) return true
            }
        }
        return flips >= 2
    }

    private fun noVowels(word: String): Boolean {
        val letters = word.filter(Char::isLetter)
        if (letters.length < 6 || letters.length != word.count { !it.isDigit() && it != '-' && it != '.' }) return false
        if (!letters.all { Character.UnicodeScript.of(it.code) == Character.UnicodeScript.LATIN }) return false
        return letters.lowercase().none { it in VOWELS }
    }
}
