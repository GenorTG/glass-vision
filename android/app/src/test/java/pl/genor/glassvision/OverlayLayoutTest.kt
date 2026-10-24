package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The in-place translation's math (OverlayLayout): colours, contrast, font fitting, hit testing. */
class OverlayLayoutTest {
    /** Monospace stand-in for Paint.measureText at size 1: every character 0.5 wide. */
    private val mono: (String) -> Float = { it.length * 0.5f }

    @Test
    fun contrast_follows_wcag() {
        assertEquals(21.0, OverlayLayout.contrast(OverlayLayout.BLACK, OverlayLayout.WHITE), 0.01)
        assertEquals(1.0, OverlayLayout.contrast(0xFF336699.toInt(), 0xFF336699.toInt()), 1e-9)
        assertEquals(0.2126, OverlayLayout.luminance(0xFFFF0000.toInt()), 1e-4)
    }

    @Test
    fun text_colour_keeps_the_original_only_when_legible() {
        val green = 0xFF1B7F3B.toInt() // a green exit sign
        assertEquals(OverlayLayout.WHITE, OverlayLayout.textColor(green, 0xFFFFFFFF.toInt()))
        // Sampled glyphs barely differ from the background (blurred, anti-aliased): black or white instead.
        assertEquals(OverlayLayout.WHITE, OverlayLayout.textColor(green, 0xFF2A8C4A.toInt()))
        assertEquals(OverlayLayout.BLACK, OverlayLayout.textColor(0xFFF5F0E0.toInt(), null)) // paper
        assertEquals(OverlayLayout.WHITE, OverlayLayout.textColor(0xFF102A5C.toInt(), null)) // navy board
        val yellowText = 0xFFFFD400.toInt()
        assertEquals(yellowText, OverlayLayout.textColor(0xFF101010.toInt(), yellowText)) // keeps a legible colour
    }

    @Test
    fun background_and_glyph_colours_are_medians() {
        val bg = 0xFF1B7F3B.toInt()
        val ring = List(40) { bg } + List(5) { 0xFFFFFFFF.toInt() } // a few glyph pixels leak into the ring
        assertEquals(bg, OverlayLayout.median(ring))
        assertNull(OverlayLayout.median(emptyList()))
        val inside = List(70) { bg } + List(30) { 0xFFF8F8F8.toInt() }
        assertEquals(0xFFF8F8F8.toInt(), OverlayLayout.glyphColor(inside, bg))
        assertNull(OverlayLayout.glyphColor(List(100) { bg } + listOf(OverlayLayout.WHITE), bg)) // too few glyph pixels
    }

    @Test
    fun text_fits_at_the_largest_size() {
        // One line "Exit" (2.0 wide at size 1) in a 100×30 box, line height 1.2: height limits it to 25.
        val one = OverlayLayout.fit("Exit", 100f, 30f, mono, 1.2f, minSize = 6f, maxSize = 200f)
        assertEquals(25f, one.size, 0.01f)
        assertEquals(listOf("Exit"), one.lines)
        assertTrue(one.fits)
        // Width limits a long line, until wrapping into two lines fits better.
        val two = OverlayLayout.fit("Emergency exit", 140f, 60f, mono, 1.2f, minSize = 6f, maxSize = 200f)
        assertEquals(listOf("Emergency", "exit"), two.lines)
        assertEquals(25f, two.size, 0.1f) // 2 lines × 1.2 × 25 = 60; "Emergency" = 4.5 × 25 = 112.5 ≤ 140
        // Forced line breaks of a list stay, each paragraph wraps on its own.
        val list = OverlayLayout.fit("Exit\nPlatform 3", 200f, 100f, mono, 1.2f, minSize = 6f, maxSize = 30f)
        assertEquals(listOf("Exit", "Platform 3"), list.lines)
        assertEquals(30f, list.size, 0.01f) // capped at maxSize (the original text height)
    }

    @Test
    fun text_that_never_fits_is_set_at_the_minimum() {
        val f = OverlayLayout.fit("Notausgangstreppenhaus", 20f, 10f, mono, 1.2f, minSize = 6f, maxSize = 50f)
        assertFalse(f.fits)
        assertEquals(6f, f.size)
        assertEquals(listOf("Notausgangstreppenhaus"), f.lines)
        assertEquals(emptyList<String>(), OverlayLayout.fit(" ", 100f, 100f, mono, 1.2f, 6f, 50f).lines)
    }

    @Test
    fun cjk_wraps_between_characters() {
        val f = OverlayLayout.fit("出口在左边", 25f, 100f, mono, 1.2f, minSize = 6f, maxSize = 20f)
        assertEquals(listOf("出口", "在左", "边"), f.lines) // at the 20 cap, 2 characters (2 × 0.5 × 20 = 20) per line
        assertEquals(20f, f.size)
    }

    @Test
    fun taps_hit_the_smallest_block_under_the_finger() {
        val tilted = Quad(listOf(Pt(0f, 0f), Pt(80f, 60f), Pt(68f, 76f), Pt(-12f, 16f)))
        assertTrue(OverlayLayout.contains(tilted.points, 40f, 38f))
        assertFalse(OverlayLayout.contains(tilted.points, 70f, 10f)) // inside the box, outside the tilted quad
        val big = Quad.of(TextBox(0f, 0f, 200f, 200f))
        val small = Quad.of(TextBox(50f, 50f, 100f, 80f))
        assertEquals(1, OverlayLayout.hit(listOf(big, small), 60f, 60f))
        assertEquals(0, OverlayLayout.hit(listOf(big, small), 150f, 150f))
        assertEquals(-1, OverlayLayout.hit(listOf(big, small), 250f, 10f))
    }
}
