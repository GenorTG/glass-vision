package pl.genor.glassvision

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.util.Base64

class JpegExifTest {
    /** A real JPEG (OpenCV-written JFIF: SOI, APP0, …): 40×30, left half dark. */
    private val jfif: ByteArray = Base64.getDecoder().decode(
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1RV19i" +
            "Z2hnPk1xeXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2Nj" +
            "Y2NjY2NjY2P/wAARCAAeACgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUF" +
            "BAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVW" +
            "V1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi" +
            "4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAEC" +
            "AxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVm" +
            "Z2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq" +
            "8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDj6KKKAOw8Af8AL/8A9s//AGauwrj/AAB/y/8A/bP/ANmrsKACiiigDx+iiigDsPAH/L//" +
            "ANs//Zq7CuP8Af8AL/8A9s//AGauwoAKKKKAP//Z",
    )

    /** [jpeg] without its APP0 (cameras write SOI + APP1 Exif first). */
    private fun withoutApp0(jpeg: ByteArray): ByteArray {
        check(jpeg[2] == 0xFF.toByte() && jpeg[3] == 0xE0.toByte())
        val len = ((jpeg[4].toInt() and 0xFF) shl 8) or (jpeg[5].toInt() and 0xFF)
        return jpeg.copyOfRange(0, 2) + jpeg.copyOfRange(4 + len, jpeg.size)
    }

    /** A camera-like Exif APP1: [little]-endian TIFF, IFD0 with Make + (optionally) Orientation. */
    private fun cameraExif(little: Boolean, orientation: Int?): ByteArray {
        val entries = mutableListOf<IntArray>() // tag, type, count, value
        entries += intArrayOf(0x010F, 2, 4, 0x47560000) // Make "GV\0\0" inline (ASCII, 4 bytes)
        if (orientation != null) entries += intArrayOf(0x0112, 3, 1, orientation)
        val tiff = ByteArrayOutputStream()
        fun u16(v: Int) = if (little) { tiff.write(v and 0xFF); tiff.write(v shr 8 and 0xFF) } else { tiff.write(v shr 8 and 0xFF); tiff.write(v and 0xFF) }
        fun u32(v: Int) = if (little) { u16(v and 0xFFFF); u16(v ushr 16) } else { u16(v ushr 16); u16(v and 0xFFFF) }
        tiff.write(if (little) 'I'.code else 'M'.code); tiff.write(if (little) 'I'.code else 'M'.code)
        u16(42); u32(8)
        u16(entries.size)
        for ((tag, type, count, value) in entries) {
            u16(tag); u16(type); u32(count)
            when (type) {
                3 -> { u16(value); u16(0) }
                else -> { tiff.write(value ushr 24); tiff.write(value ushr 16 and 0xFF); tiff.write(value ushr 8 and 0xFF); tiff.write(value and 0xFF) }
            }
        }
        u32(0)
        val body = "Exif\u0000\u0000".toByteArray(Charsets.ISO_8859_1) + tiff.toByteArray()
        val len = body.size + 2
        return byteArrayOf(0xFF.toByte(), 0xE1.toByte(), (len shr 8).toByte(), len.toByte()) + body
    }

    private fun withSegment(jpeg: ByteArray, segment: ByteArray) = jpeg.copyOfRange(0, 2) + segment + jpeg.copyOfRange(2, jpeg.size)

    /** javax.imageio decodes it (by reflection: the unit-test compile classpath is android.jar, not java.desktop). */
    private fun decodes(jpeg: ByteArray) {
        val img = Class.forName("javax.imageio.ImageIO").getMethod("read", InputStream::class.java)
            .invoke(null, ByteArrayInputStream(jpeg))
        assertNotNull("not a decodable JPEG", img)
        assertEquals(40, img.javaClass.getMethod("getWidth").invoke(img))
        assertEquals(30, img.javaClass.getMethod("getHeight").invoke(img))
    }

    @Test
    fun degrees_and_orientation_values() {
        assertEquals(1, JpegExif.orientationFor(0))
        assertEquals(6, JpegExif.orientationFor(90))
        assertEquals(3, JpegExif.orientationFor(180))
        assertEquals(8, JpegExif.orientationFor(270))
        assertEquals(8, JpegExif.orientationFor(-90))
        for (d in listOf(0, 90, 180, 270)) assertEquals(d, JpegExif.degreesOf(JpegExif.orientationFor(d)))
    }

    @Test
    fun adds_orientation_to_a_jpeg_without_exif() {
        assertNull(JpegExif.orientation(jfif))
        val out = JpegExif.withOrientation(jfif, 6)
        assertEquals(6, JpegExif.orientation(out))
        // JFIF's APP0 stays first, the image data is untouched.
        assertEquals(0xE0.toByte(), out[3])
        decodes(out)
        val bare = withoutApp0(jfif)
        val out2 = JpegExif.withOrientation(bare, 8)
        assertEquals(8, JpegExif.orientation(out2))
        assertEquals(0xE1.toByte(), out2[3])
        assertArrayEquals(bare.copyOfRange(2, bare.size), out2.copyOfRange(2 + 36, out2.size))
    }

    @Test
    fun patches_a_camera_exif_orientation_in_place_in_both_byte_orders() {
        for (little in listOf(true, false)) {
            val camera = withSegment(withoutApp0(jfif), cameraExif(little, orientation = 1))
            assertEquals(1, JpegExif.orientation(camera))
            val out = JpegExif.withOrientation(camera, 3)
            assertEquals(3, JpegExif.orientation(out))
            assertEquals(camera.size, out.size)
            assertEquals(1, camera.indices.count { camera[it] != out[it] })
            assertEquals(1, JpegExif.orientation(camera)) // input left as is
            decodes(out)
        }
    }

    @Test
    fun replaces_a_camera_exif_without_orientation() {
        val camera = withSegment(withoutApp0(jfif), cameraExif(little = true, orientation = null))
        assertNull(JpegExif.orientation(camera))
        val out = JpegExif.withOrientation(camera, 6)
        assertEquals(6, JpegExif.orientation(out))
        // One Exif segment: the old one is gone.
        val exifs = (0 until out.size - 6).count { i ->
            out[i] == 0xFF.toByte() && out[i + 1] == 0xE1.toByte() && String(out, i + 4, 4, Charsets.ISO_8859_1) == "Exif"
        }
        assertEquals(1, exifs)
        decodes(out)
    }

    @Test
    fun not_a_jpeg() {
        assertNull(JpegExif.orientation(byteArrayOf(1, 2, 3, 4, 5)))
        assertNull(JpegExif.orientation(ByteArray(0)))
    }

    /** With GV_EXIF_SAMPLES=<dir>: writes samples for checking with another reader (OpenCV, as on the server). */
    @Test
    fun writes_samples_for_external_readers() {
        val dir = System.getenv("GV_EXIF_SAMPLES")?.let(::File) ?: return
        dir.mkdirs()
        File(dir, "plain.jpg").writeBytes(jfif)
        for (o in listOf(1, 3, 6, 8)) {
            File(dir, "jfif_o$o.jpg").writeBytes(JpegExif.withOrientation(jfif, o))
            File(dir, "camera_ii_o$o.jpg").writeBytes(
                JpegExif.withOrientation(withSegment(withoutApp0(jfif), cameraExif(true, 1)), o),
            )
            File(dir, "camera_mm_noori_o$o.jpg").writeBytes(
                JpegExif.withOrientation(withSegment(withoutApp0(jfif), cameraExif(false, null)), o),
            )
        }
    }
}
