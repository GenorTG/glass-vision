package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import kotlin.random.Random

class StillQualityTest {
    /** Text-like luma: dark strokes of varied width on a light ground, deterministic. */
    private fun strokes(w: Int, h: Int, ink: Int = 30, paper: Int = 220): IntArray {
        val rnd = Random(7)
        val out = IntArray(w * h) { paper }
        repeat(w * h / 60) {
            val x = rnd.nextInt(w - 6)
            val y = rnd.nextInt(h - 12)
            val vertical = rnd.nextBoolean()
            for (k in 0 until 10) for (t in 0 until 2) {
                val (px, py) = if (vertical) (x + t) to (y + k) else (x + k / 2) to (y + t)
                out[py * w + px] = ink
            }
        }
        return out
    }

    /** Separable box blur of radius [r] (a defocused still). */
    private fun blur(src: IntArray, w: Int, h: Int, r: Int): IntArray {
        val tmp = IntArray(src.size)
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0
            var n = 0
            for (d in -r..r) { val xx = x + d; if (xx in 0 until w) { s += src[y * w + xx]; n++ } }
            tmp[y * w + x] = s / n
        }
        val out = IntArray(src.size)
        for (y in 0 until h) for (x in 0 until w) {
            var s = 0
            var n = 0
            for (d in -r..r) { val yy = y + d; if (yy in 0 until h) { s += tmp[yy * w + x]; n++ } }
            out[y * w + x] = s / n
        }
        return out
    }

    /** Horizontal motion blur over [len] pixels (shake at the shutter). */
    private fun motion(src: IntArray, w: Int, h: Int, len: Int): IntArray = IntArray(src.size) { i ->
        val x = i % w
        val y = i / w
        var s = 0
        var n = 0
        for (d in 0 until len) { val xx = x + d - len / 2; if (xx in 0 until w) { s += src[y * w + xx]; n++ } }
        s / n
    }

    @Test
    fun blur_lowers_sharpness_a_lot() {
        val w = 200
        val h = 150
        val sharp = strokes(w, h)
        val s = Sharpness.of(sharp, w, h)
        assertTrue(s > 0f)
        assertTrue(Sharpness.of(blur(sharp, w, h, 1), w, h) < s * 0.5f)
        assertTrue(Sharpness.of(blur(sharp, w, h, 3), w, h) < s * 0.1f)
        assertTrue(Sharpness.of(motion(sharp, w, h, 9), w, h) < s * 0.5f)
    }

    @Test
    fun sharpness_ignores_contrast_and_brightness() {
        val w = 120
        val h = 90
        val bright = Sharpness.of(strokes(w, h, ink = 30, paper = 230), w, h)
        val dim = Sharpness.of(strokes(w, h, ink = 10, paper = 110), w, h)
        assertEquals(bright, dim, bright * 0.02f)
    }

    @Test
    fun flat_or_tiny_regions_measure_zero() {
        assertEquals(0f, Sharpness.of(IntArray(100) { 128 }, 10, 10))
        assertEquals(0f, Sharpness.of(IntArray(4) { it * 60 }, 2, 2))
    }

    @Test
    fun luma_of_argb_and_rgba() {
        val argb = intArrayOf(0xFFFFFFFF.toInt(), 0xFF000000.toInt(), 0xFFFF0000.toInt(), 0xFF00FF00.toInt())
        assertEquals(listOf(255, 0, 76, 149), Sharpness.lumaOfArgb(argb).toList())

        // 3×2 RGBA plane with 4 bytes of row padding: rect (1,0)-(3,2).
        val rowStride = 3 * 4 + 4
        val buf = ByteBuffer.allocate(rowStride * 2)
        fun put(x: Int, y: Int, r: Int, g: Int, b: Int) {
            val at = y * rowStride + x * 4
            buf.put(at, r.toByte()); buf.put(at + 1, g.toByte()); buf.put(at + 2, b.toByte()); buf.put(at + 3, -1)
        }
        put(0, 0, 0, 0, 0); put(1, 0, 255, 255, 255); put(2, 0, 255, 0, 0)
        put(0, 1, 9, 9, 9); put(1, 1, 0, 255, 0); put(2, 1, 0, 0, 255)
        val (luma, w, h) = Sharpness.lumaOfRgba(buf, rowStride, 4, PixelRect(1, 0, 3, 2))
        assertEquals(2, w)
        assertEquals(2, h)
        assertEquals(listOf(255, 76, 149, 28), luma.toList())
        val (sub, sw, sh) = Sharpness.lumaOfRgba(buf, rowStride, 4, PixelRect(0, 0, 3, 2), step = 2)
        assertEquals(2 to 1, sw to sh)
        assertEquals(listOf(0, 76), sub.toList())
    }

    @Test
    fun rotating_a_region_with_its_picture() {
        val r = NormRect(0.1f, 0.2f, 0.4f, 0.3f)
        // Clockwise 90: the top-left corner moves to the top-right.
        assertRect(NormRect(0.7f, 0.1f, 0.8f, 0.4f), r.rotated(90))
        assertRect(NormRect(0.6f, 0.7f, 0.9f, 0.8f), r.rotated(180))
        assertRect(NormRect(0.2f, 0.6f, 0.3f, 0.9f), r.rotated(270))
        assertRect(r.rotated(270), r.rotated(-90))
        for (d in listOf(0, 90, 180, 270)) assertRect(r, r.rotated(d).rotated(-d))
    }

    @Test
    fun region_to_pixels_is_clamped_and_never_empty() {
        assertEquals(PixelRect(100, 200, 400, 300), NormRect(0.1f, 0.2f, 0.4f, 0.3f).toPixels(1000, 1000))
        assertEquals(PixelRect(0, 0, 10, 10), NormRect(-1f, -1f, 2f, 2f).toPixels(10, 10))
        assertEquals(PixelRect(9, 9, 10, 10), NormRect(1f, 1f, 1f, 1f).toPixels(10, 10))
        assertEquals(PixelRect(0, 0, 5, 5), NormRect(0.5f, 0.5f, 0f, 0f).toPixels(10, 10))
    }

    @Test
    fun text_marks_are_text_detections_largest_first() {
        fun det(pipeline: String, box: NormBox, seen: Int = 0) = Detection(pipeline, "x", 0.9f, box, "$pipeline:1", seen)
        val small = det("text", NormBox(0.1f, 0.1f, 0.2f, 0.15f))
        val large = det("text", NormBox(0.3f, 0.5f, 0.9f, 0.7f))
        val tiny = det("text", NormBox(0.5f, 0.5f, 0.51f, 0.51f))
        val held = det("text", NormBox(0.05f, 0.8f, 0.25f, 0.9f), seen = 5000) // slow OCR lane: held, still shown
        val person = det("objects", NormBox(0f, 0f, 1f, 1f))
        val result = ServerMessage.Detections(1, 0, 720, 960, 0.0, 0.0, emptyMap(), emptyList(), listOf(small, person, tiny, held, large))
        val marks = TextMarks.of(result)
        assertEquals(listOf(NormRect.of(large.box), NormRect.of(held.box), NormRect.of(small.box)), marks)
        assertEquals(NormRect(0.05f, 0.1f, 0.9f, 0.9f), TextMarks.union(marks))
        assertNull(TextMarks.union(emptyList()))
        assertEquals(emptyList<NormRect>(), TextMarks.of(null))
        // Raw detections when the server sends no tracks (smoothing off).
        val raw = result.copy(dets = listOf(large), tracks = emptyList())
        assertEquals(listOf(NormRect.of(large.box)), TextMarks.of(raw))
    }

    @Test
    fun blur_check_compares_with_the_best_live_frame() {
        assertNull(BlurCheck.ratio(1f, null))
        assertNull(BlurCheck.ratio(1f, 0f))
        assertNull(BlurCheck.ratio(0f, 1f))
        assertEquals(2f, BlurCheck.ratio(0.4f, 0.2f)!!, 1e-6f)
        assertTrue(BlurCheck.isBlurry(0.09f, 0.2f))
        assertFalse(BlurCheck.isBlurry(0.11f, 0.2f))
        assertFalse(BlurCheck.isBlurry(0.01f, null))
    }

    @Test
    fun blurred_still_fails_the_check_against_a_sharp_live_frame() {
        val w = 160
        val h = 120
        val live = Sharpness.of(blur(strokes(w, h), w, h, 0), w, h) // r = 0: identity
        val sharpStill = Sharpness.of(strokes(w, h), w, h)
        val shaken = Sharpness.of(motion(strokes(w, h), w, h, 7), w, h)
        assertFalse(BlurCheck.isBlurry(sharpStill, live))
        assertTrue(BlurCheck.isBlurry(shaken, live))
    }

    @Test
    fun sample_size_keeps_at_least_the_wanted_size() {
        assertEquals(1, Still.sampleSize(4000, 2560))
        assertEquals(1, Still.sampleSize(4000, 4000))
        assertEquals(2, Still.sampleSize(4000, 2000))
        assertEquals(4, Still.sampleSize(4000, 720))
        assertEquals(1, Still.sampleSize(4000, Int.MAX_VALUE))
        assertEquals(8, Still.sampleSize(4000, 400))
    }

    private fun assertRect(expected: NormRect, actual: NormRect) {
        val e = listOf(expected.left, expected.top, expected.right, expected.bottom)
        val a = listOf(actual.left, actual.top, actual.right, actual.bottom)
        e.zip(a).forEach { (x, y) -> assertEquals("$expected vs $actual", x, y, 1e-5f) }
    }
}
