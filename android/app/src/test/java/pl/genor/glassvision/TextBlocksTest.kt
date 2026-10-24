package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

/**
 * Lines → blocks on the phone (TextBlocks, docs/research/ocr-phone-pipeline.md §4) on rendered-geometry fixtures:
 * lines as ML Kit reports them (quads clockwise from the text's top-left, pixels), tilted, in columns, RTL…
 */
class TextBlocksTest {
    /** A line drawn at ([x], [y]) [w]×[h] in the sign's own frame, the whole scene rotated [deg]° clockwise about (cx, cy). */
    private fun line(text: String, x: Float, y: Float, w: Float, h: Float, block: Int, deg: Float = 0f, conf: Float = 0.9f, cx: Float = 640f, cy: Float = 480f): OcrLine {
        val a = Math.toRadians(deg.toDouble())
        fun rot(px: Float, py: Float): Pt {
            val dx = px - cx
            val dy = py - cy
            return Pt((cx + dx * cos(a) - dy * sin(a)).toFloat(), (cy + dx * sin(a) + dy * cos(a)).toFloat())
        }
        val quad = Quad(listOf(rot(x, y), rot(x + w, y), rot(x + w, y + h), rot(x, y + h)))
        return OcrLine(text, conf, quad.bounds(), block, quad = quad)
    }

    private fun blocksOf(g: OcrGrouping): List<List<String>> = g.blocks.map { b -> b.lines.map { g.lines[it].text } }

    @Test
    fun two_signs_side_by_side_read_sign_by_sign() {
        // ML Kit made every line its own block (as on a rendered two-sign image); the right sign starts higher.
        val g = TextBlocks.group(
            listOf(
                line("LIFT", 720f, 246f, 180f, 64f, 3),
                line("TOILETS", 720f, 120f, 330f, 70f, 1),
                line("EXIT", 80f, 130f, 220f, 70f, 0),
                line("GATE 42", 80f, 256f, 380f, 64f, 2),
            )
        )
        assertEquals(listOf(listOf("EXIT", "GATE 42"), listOf("TOILETS", "LIFT")), blocksOf(g))
        assertEquals(listOf(0, 0, 1, 1), g.lines.map { it.block })
        assertEquals(0f, g.angle, 0.01f)
    }

    @Test
    fun tilted_signs_are_deskewed_before_grouping() {
        // The same two 3-line signs, photographed 25° clockwise: their axis-aligned boxes overlap and interleave.
        for (deg in listOf(-25f, 25f)) {
            val lines = listOf(
                line("NOTAUSGANG", 100f, 300f, 400f, 60f, 0, deg),
                line("Bitte freihalten", 100f, 375f, 420f, 50f, 1, deg),
                line("Tag und Nacht", 100f, 440f, 360f, 50f, 2, deg),
                line("KASSE", 700f, 300f, 260f, 60f, 3, deg),
                line("Eingang links", 700f, 375f, 340f, 50f, 4, deg),
                line("Ausgang rechts", 700f, 440f, 360f, 50f, 5, deg),
            ).shuffled(java.util.Random(7))
            val g = TextBlocks.group(lines)
            assertEquals(deg, g.angle, 0.5f)
            assertEquals(
                "$deg°",
                listOf(listOf("NOTAUSGANG", "Bitte freihalten", "Tag und Nacht"), listOf("KASSE", "Eingang links", "Ausgang rechts")),
                blocksOf(g),
            )
            // The block quad is the tilted union, not an axis-aligned box: its angle is the sign's.
            assertEquals(deg, g.blocks[0].quad.angle, 0.5f)
            assertEquals(420f, g.blocks[0].quad.length, 1f)
        }
    }

    @Test
    fun upside_down_sign_reads_top_line_first() {
        val g = TextBlocks.group(listOf(line("ERSTE", 100f, 300f, 300f, 60f, 0, 180f), line("ZWEITE", 100f, 380f, 300f, 60f, 1, 180f)))
        assertEquals(180f, kotlin.math.abs(g.angle), 0.5f)
        assertEquals(listOf(listOf("ERSTE", "ZWEITE")), blocksOf(g))
    }

    @Test
    fun two_columns_of_prose_are_two_blocks_in_column_order() {
        val left = listOf("Der Zug nach Berlin", "fährt heute von einem", "anderen Gleis ab als", "im Fahrplan steht.")
        val right = listOf("The train to Berlin", "leaves from another", "platform today than", "the timetable says.")
        val lines = left.mapIndexed { i, t -> line(t, 60f, 100f + i * 44f, 420f - (if (i == 3) 120f else 0f), 34f, 0) } +
            right.mapIndexed { i, t -> line(t, 560f, 100f + i * 44f, 410f - (if (i == 3) 90f else 0f), 34f, 1) }
        val g = TextBlocks.group(lines.reversed())
        assertEquals(listOf(left, right), blocksOf(g))
        assertEquals(listOf(BlockKind.PROSE, BlockKind.PROSE), g.blocks.map { it.kind })
    }

    @Test
    fun heading_and_body_are_separate_blocks() {
        val g = TextBlocks.group(
            listOf(
                line("Öffnungszeiten", 100f, 100f, 520f, 64f, 0),
                line("Montag bis Freitag von acht", 100f, 180f, 500f, 32f, 0), // ML Kit put the heading in the body's block
                line("bis achtzehn Uhr geöffnet", 100f, 220f, 470f, 32f, 0),
            )
        )
        assertEquals(listOf(listOf("Öffnungszeiten"), listOf("Montag bis Freitag von acht", "bis achtzehn Uhr geöffnet")), blocksOf(g))
        assertEquals(BlockKind.PROSE, g.blocks[1].kind)
    }

    @Test
    fun row_pieces_are_stitched_but_column_gutters_are_not() {
        val g = TextBlocks.group(
            listOf(
                line("12", 330f, 100f, 80f, 60f, 1),
                line("Gate", 100f, 100f, 180f, 60f, 0), // gap 50 = 0.83 h: one line
                line("Kasse", 900f, 104f, 200f, 56f, 2), // gap 490: another sign
            )
        )
        assertEquals(listOf(listOf("Gate 12"), listOf("Kasse")), blocksOf(g))
        val stitched = g.lines[0]
        assertEquals(100f, stitched.box.left, 0.5f)
        assertEquals(410f, stitched.box.right, 0.5f)
    }

    @Test
    fun rtl_rows_and_blocks_go_right_to_left() {
        // Two pieces of one Arabic row ("مخرج" right, "الطوارئ" left), and a second Arabic sign further left.
        val g = TextBlocks.group(
            listOf(
                line("الطوارئ", 500f, 100f, 200f, 60f, 1),
                line("مخرج", 740f, 100f, 160f, 60f, 0),
                line("مدخل", 60f, 100f, 160f, 60f, 2),
            )
        )
        assertEquals(listOf(listOf("مخرج الطوارئ"), listOf("مدخل")), blocksOf(g))
        assertTrue(TextBlocks.isRtl("مخرج"))
        assertTrue(TextBlocks.isRtl("יציאה 3"))
        assertFalse(TextBlocks.isRtl("EXIT مخرج EXIT"))
    }

    @Test
    fun stacked_lines_join_only_when_near_aligned_and_alike() {
        val g = TextBlocks.group(
            listOf(
                line("NOTAUSGANG", 100f, 100f, 400f, 60f, 0),
                line("Bitte freihalten", 100f, 200f, 420f, 50f, 1), // gap 40 ≤ 0.8 × 50: same sign
                line("small print", 100f, 165f, 200f, 15f, 3), // much smaller: its own block, doesn't separate the others
                line("Etage 2", 100f, 400f, 200f, 50f, 2), // gap 150: another sign
                line("Kasse", 900f, 100f, 200f, 60f, 4), // no horizontal relation
            )
        )
        val byText = g.lines.associate { it.text to it.block }
        assertEquals(byText["NOTAUSGANG"], byText["Bitte freihalten"])
        assertEquals(4, g.blocks.size)
        // A line of comparable size between two (another sign's line) keeps them apart, even inside one engine block.
        val between = TextBlocks.group(
            listOf(line("Abc def", 100f, 100f, 300f, 40f, 0), line("Ghi", 100f, 146f, 200f, 26f, 0), line("Jkl mno", 100f, 180f, 300f, 40f, 0))
        )
        assertEquals(listOf(listOf("Abc def"), listOf("Ghi"), listOf("Jkl mno")), blocksOf(between))
    }

    @Test
    fun prose_or_list() {
        fun kind(vararg lines: Pair<String, Float>) = TextBlocks.kind(lines.map { it.first }, lines.map { it.second }, lines.maxOf { it.second })
        assertEquals(BlockKind.PROSE, kind("Notausgang" to 300f))
        assertEquals(BlockKind.LIST, kind("Kaffee 2,50" to 300f, "Tee 2,00" to 250f)) // prices
        assertEquals(BlockKind.LIST, kind("Öffnungszeiten:" to 300f, "Mo–Fr 8–18" to 280f))
        assertEquals(BlockKind.LIST, kind("EXIT" to 200f, "GATE 42" to 300f))
        assertEquals(BlockKind.LIST, kind("TOILETTEN" to 300f, "AUFZUG" to 290f)) // one word per line: a sign stack
        assertEquals(BlockKind.PROSE, kind("Bitte nicht" to 300f, "rauchen" to 200f)) // lowercase continuation
        assertEquals(BlockKind.PROSE, kind("Wir bitten um Ihr Ver-" to 400f, "ständnis." to 120f))
        assertEquals(BlockKind.LIST, kind("Gleis 1 Berlin Hbf" to 400f, "Gleis Hamburg" to 200f, "Gleis Köln Messe" to 380f)) // short middle line
        assertEquals(BlockKind.PROSE, kind("Please keep this door" to 400f, "Closed at all times" to 380f))
    }

    @Test
    fun prose_is_joined_and_dehyphenated() {
        assertEquals("Wir bitten um Ihr Verständnis.", TextBlocks.joinProse(listOf("Wir bitten um Ihr Ver-", "ständnis.")))
        assertEquals("Ein- und Ausgang", TextBlocks.joinProse(listOf("Ein-", "und Ausgang"))) // suspended hyphen
        assertEquals("Nord- Süd", TextBlocks.joinProse(listOf("Nord-", "Süd")))
        assertEquals("出口在左边", TextBlocks.joinProse(listOf("出口", "在左边")))
        assertEquals("a b", TextBlocks.joinProse(listOf(" a ", "", "b")))
    }

    @Test
    fun skew_is_a_weighted_median_and_wraps_at_180() {
        val lines = listOf(5f, 6f, 4f, 40f).map { line("x", 100f, 100f, 300f, 40f, 0, it) }
        assertEquals(5f, TextBlocks.skew(lines), 1f) // the 40° outlier doesn't pull it
        val flipped = listOf(179f, -179f, 178f).map { line("x", 100f, 100f, 300f, 40f, 0, it) }
        assertEquals(179f, kotlin.math.abs(TextBlocks.skew(flipped)), 1.5f)
        assertEquals(0f, TextBlocks.skew(listOf(OcrLine("box only", 0.9f, TextBox(0f, 0f, 10f, 5f), 0))), 0f)
    }

    @Test
    fun blank_lines_are_dropped() {
        val g = TextBlocks.group(listOf(line(" ", 0f, 0f, 10f, 10f, 0), line("a", 0f, 100f, 100f, 30f, 1)))
        assertEquals(listOf(listOf("a")), blocksOf(g))
        assertEquals(OcrGrouping(emptyList(), emptyList(), 0f), TextBlocks.group(emptyList()))
    }

    @Test
    fun local_reading_normalizes_and_flags_suspect_blocks() {
        val g = TextBlocks.group(
            listOf(
                line("Ausgang", 100f, 100f, 300f, 60f, 0),
                line("Notausgang 2", 100f, 170f, 420f, 50f, 0),
                line("BbIXOI", 100f, 500f, 300f, 60f, 1, conf = 0.55f),
            )
        )
        val r = LocalReading.of(g, listOf("de", "de", null), 1280f, 960f)
        assertEquals(listOf("Ausgang", "Notausgang 2", "BbIXOI"), r.lines.map { it.text })
        assertEquals(listOf(false, false, true), r.lines.map { it.suspect })
        assertEquals(listOf(false, true), r.blocks.map { it.suspect })
        assertEquals(listOf("de", null), r.blocks.map { it.lang })
        assertEquals("Ausgang\nNotausgang 2", r.blocks[0].text)
        assertEquals(BlockKind.LIST, r.blocks[0].kind)
        val box = r.blocks[0].box
        assertEquals(100f / 1280f, box.left, 1e-4f)
        assertEquals(520f / 1280f, box.right, 1e-4f)
        assertEquals(220f / 960f, box.bottom, 1e-4f)
        assertTrue(r.lines.all { it.quad != null })
    }

    @Test
    fun suspect_readings() {
        assertFalse(TextSuspect.line("EXIT 12", 0.83f))
        assertFalse(TextSuspect.line("iPhone 15 Pro", 0.9f))
        assertFalse(TextSuspect.line("McDonald's", 0.9f))
        assertFalse(TextSuspect.line("Ausgang", 0f)) // unknown confidence (unbundled ML Kit model): no penalty
        assertFalse(TextSuspect.line("出口 / でぐち / 出口です", 0.9f)) // Han + kana is Japanese, not mixing
        assertFalse(TextSuspect.line("12:45", 0.6f))
        assertFalse(TextSuspect.line("Wyjście awaryjne", 0.45f)) // only unsure
        assertTrue(TextSuspect.line("BbIXOI", 0.55f)) // ВЫХОД read by ML Kit Latin
        assertTrue(TextSuspect.line("\u0415\u0425I\u0422", 0.9f)) // Cyrillic Е, Х, Т with a Latin I in one word
        assertTrue(TextSuspect.line("c3r0 em1s1ones", 0.45f))
        assertTrue(TextSuspect.line("%&#/ ~~", 0.4f))
        assertTrue(TextSuspect.line("anything", 0.2f)) // very unsure alone
        assertTrue(TextSuspect.line("Wrtzklmn", 0.48f))
        // A block is suspect when its suspect lines hold half its letters.
        assertTrue(TextSuspect.block(listOf("BbIXOI" to true, "EXIT" to false)))
        assertFalse(TextSuspect.block(listOf("BbIXOI" to true, "Notausgang 2" to false)))
        assertFalse(TextSuspect.block(emptyList()))
    }
}
