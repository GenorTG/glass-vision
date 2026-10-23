package pl.genor.glassvision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import app.cash.paparazzi.Paparazzi
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Rule
import org.junit.Test

/**
 * [FrameEncoder.uprightBitmap] on real Bitmaps (the Paparazzi rule loads layoutlib's native graphics): the
 * sensor image rotated clockwise by CameraX's rotationDegrees comes out upright, for the back camera (90°)
 * and the front one (usually 270°), and is never mirrored.
 */
class FrameRotationTest {
    @get:Rule
    val paparazzi = Paparazzi()

    /**
     * Landscape sensor image 80×60 with coloured quadrants. Seen from the sensor: red top-left, green
     * top-right, blue bottom-left, white bottom-right.
     */
    private fun sensorImage(): Bitmap = Bitmap.createBitmap(80, 60, Bitmap.Config.ARGB_8888).also { bmp ->
        val c = Canvas(bmp)
        val p = Paint()
        fun quad(color: Int, l: Float, t: Float) {
            p.color = color
            c.drawRect(l, t, l + 40f, t + 30f, p)
        }
        quad(Color.RED, 0f, 0f)
        quad(Color.GREEN, 40f, 0f)
        quad(Color.BLUE, 0f, 30f)
        quad(Color.WHITE, 40f, 30f)
    }

    /** Colour at the centre of each quadrant: [top-left, top-right, bottom-left, bottom-right]. */
    private fun quadrants(bmp: Bitmap): List<Int> {
        val (w, h) = bmp.width to bmp.height
        return listOf(
            bmp.getPixel(w / 4, h / 4), bmp.getPixel(3 * w / 4, h / 4),
            bmp.getPixel(w / 4, 3 * h / 4), bmp.getPixel(3 * w / 4, 3 * h / 4),
        )
    }

    @Test
    fun rotation_0_is_untouched() {
        val src = sensorImage()
        assertSame(src, FrameEncoder.uprightBitmap(src, 0, 960))
    }

    @Test
    fun back_camera_90_rotates_clockwise() {
        val out = FrameEncoder.uprightBitmap(sensorImage(), 90, 960)
        assertEquals(60 to 80, out.width to out.height)
        // Clockwise: the sensor's left column becomes the top row → blue top-left, red top-right.
        assertEquals(listOf(Color.BLUE, Color.RED, Color.WHITE, Color.GREEN), quadrants(out))
    }

    @Test
    fun front_camera_270_rotates_counter_clockwise_without_mirroring() {
        val out = FrameEncoder.uprightBitmap(sensorImage(), 270, 960)
        assertEquals(60 to 80, out.width to out.height)
        // 270° clockwise = 90° counter-clockwise: the sensor's right column becomes the top row. A mirror
        // would swap left/right (green ↔ red); a rotation keeps the red→green→white→blue cycle order.
        assertEquals(listOf(Color.GREEN, Color.WHITE, Color.RED, Color.BLUE), quadrants(out))
    }

    @Test
    fun rotates_and_downscales_in_one_pass() {
        val out = FrameEncoder.uprightBitmap(sensorImage(), 270, 40)
        assertEquals(FrameEncoder.uprightScaledSize(80, 60, 270, 40).let { it.width to it.height }, out.width to out.height)
        assertEquals(30 to 40, out.width to out.height)
        assertEquals(listOf(Color.GREEN, Color.WHITE, Color.RED, Color.BLUE), quadrants(out))
    }
}
