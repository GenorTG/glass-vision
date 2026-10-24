package pl.genor.glassvision

/**
 * The EXIF Orientation tag of an in-memory JPEG, read and written without decoding the image: the Read still goes
 * to the server as the camera's own JPEG plus this tag (the server applies it, docs/PROTOCOL.md), instead of a
 * decode → rotate → re-encode of 12 MP on the phone.
 *
 * Writing patches an existing IFD0 Orientation entry in place; otherwise the camera's Exif segment (if any) is replaced
 * by a minimal one holding only the orientation (inserting an entry into a foreign IFD would shift every offset in it). */
object JpegExif {
    private const val TAG_ORIENTATION = 0x0112
    private const val TYPE_SHORT = 3
    private val EXIF_ID = byteArrayOf('E'.code.toByte(), 'x'.code.toByte(), 'i'.code.toByte(), 'f'.code.toByte(), 0, 0)

    /** EXIF orientation (1 = upright) that tells a viewer to rotate the stored pixels clockwise by [degrees]. */
    fun orientationFor(degrees: Int): Int = when (Math.floorMod(degrees, 360)) {
        90 -> 6
        180 -> 3
        270 -> 8
        else -> 1
    }

    /** Clockwise degrees for an EXIF [orientation] (mirrored variants: their rotation part). */
    fun degreesOf(orientation: Int): Int = when (orientation) {
        3, 4 -> 180
        5, 6 -> 90
        7, 8 -> 270
        else -> 0
    }

    /** The IFD0 Orientation value, or null (no Exif segment / no tag / not a JPEG). */
    fun orientation(jpeg: ByteArray): Int? {
        val seg = exifSegment(jpeg) ?: return null
        val entry = orientationEntry(jpeg, seg) ?: return null
        return read16(jpeg, entry + 8, seg.littleEndian)
    }

    /** [jpeg] with its Orientation set to [orientation] (a copy; [jpeg] is left as is). */
    fun withOrientation(jpeg: ByteArray, orientation: Int): ByteArray {
        require(isJpeg(jpeg)) { "not a JPEG" }
        val seg = exifSegment(jpeg)
        if (seg != null) {
            orientationEntry(jpeg, seg)?.let { entry ->
                val out = jpeg.copyOf()
                write16(out, entry + 8, orientation, seg.littleEndian)
                return out
            }
        }
        // Our APP1 right after SOI (after a JFIF APP0, which must come first), minus the old Exif segment.
        val app1 = minimalApp1(orientation)
        val insert = jfifEnd(jpeg) ?: 2
        val out = java.io.ByteArrayOutputStream(jpeg.size + app1.size)
        out.write(jpeg, 0, insert)
        out.write(app1)
        if (seg == null) {
            out.write(jpeg, insert, jpeg.size - insert)
        } else {
            out.write(jpeg, insert, seg.start - insert)
            out.write(jpeg, seg.end, jpeg.size - seg.end)
        }
        return out.toByteArray()
    }

    /** End of a JFIF APP0 segment directly after SOI, or null. */
    private fun jfifEnd(jpeg: ByteArray): Int? {
        if (jpeg.size < 6 || jpeg[2] != 0xFF.toByte() || jpeg[3] != 0xE0.toByte()) return null
        return (4 + read16(jpeg, 4, false)).takeIf { it <= jpeg.size }
    }

    fun isJpeg(bytes: ByteArray): Boolean =
        bytes.size >= 4 && bytes[0] == 0xFF.toByte() && bytes[1] == 0xD8.toByte()

    /** APP1 "Exif" segment: [start] at its 0xFF marker, [end] after it, [tiff] = TIFF header offset. */
    private class Segment(val start: Int, val end: Int, val tiff: Int, val littleEndian: Boolean)

    private fun exifSegment(jpeg: ByteArray): Segment? {
        if (!isJpeg(jpeg)) return null
        var i = 2
        while (i + 4 <= jpeg.size) {
            if (jpeg[i] != 0xFF.toByte()) return null
            val marker = jpeg[i + 1].toInt() and 0xFF
            when {
                marker == 0xFF -> { i++; continue } // fill byte
                marker == 0xD9 || marker == 0xDA -> return null // EOI / start of scan: no more metadata
                marker in 0xD0..0xD7 || marker == 0x01 -> { i += 2; continue } // no length
            }
            val length = read16(jpeg, i + 2, false)
            if (length < 2 || i + 2 + length > jpeg.size) return null
            val body = i + 4
            if (marker == 0xE1 && length >= 2 + 6 + 8 && EXIF_ID.indices.all { jpeg[body + it] == EXIF_ID[it] }) {
                val tiff = body + 6
                val order = jpeg[tiff].toInt().toChar() to jpeg[tiff + 1].toInt().toChar()
                val little = when (order) {
                    'I' to 'I' -> true
                    'M' to 'M' -> false
                    else -> return null
                }
                return Segment(i, i + 2 + length, tiff, little)
            }
            i += 2 + length
        }
        return null
    }

    /** Offset of IFD0's Orientation entry (SHORT, inside the segment), or null. */
    private fun orientationEntry(jpeg: ByteArray, seg: Segment): Int? {
        val ifd0 = seg.tiff + read32(jpeg, seg.tiff + 4, seg.littleEndian)
        if (ifd0 + 2 > seg.end || ifd0 < seg.tiff) return null
        val count = read16(jpeg, ifd0, seg.littleEndian)
        for (n in 0 until count) {
            val entry = ifd0 + 2 + 12 * n
            if (entry + 12 > seg.end) return null
            if (read16(jpeg, entry, seg.littleEndian) == TAG_ORIENTATION) {
                val type = read16(jpeg, entry + 2, seg.littleEndian)
                return entry.takeIf { type == TYPE_SHORT }
            }
        }
        return null
    }

    /** FFE1 segment with "Exif\0\0", a big-endian TIFF header and an IFD0 holding only Orientation. */
    private fun minimalApp1(orientation: Int): ByteArray {
        val out = ByteArray(36)
        var i = 0
        fun put(vararg bytes: Int) = bytes.forEach { out[i++] = it.toByte() }
        put(0xFF, 0xE1, 0, 34)
        EXIF_ID.forEach { out[i++] = it }
        put('M'.code, 'M'.code, 0, 0x2A, 0, 0, 0, 8) // TIFF header, IFD0 at 8
        put(0, 1) // one entry
        put(0x01, 0x12, 0, TYPE_SHORT, 0, 0, 0, 1, orientation shr 8 and 0xFF, orientation and 0xFF, 0, 0)
        put(0, 0, 0, 0) // no IFD1
        check(i == out.size)
        return out
    }

    private fun read16(b: ByteArray, at: Int, little: Boolean): Int {
        val x = b[at].toInt() and 0xFF
        val y = b[at + 1].toInt() and 0xFF
        return if (little) x or (y shl 8) else (x shl 8) or y
    }

    private fun write16(b: ByteArray, at: Int, value: Int, little: Boolean) {
        val hi = (value shr 8 and 0xFF).toByte()
        val lo = (value and 0xFF).toByte()
        if (little) {
            b[at] = lo
            b[at + 1] = hi
        } else {
            b[at] = hi
            b[at + 1] = lo
        }
    }

    private fun read32(b: ByteArray, at: Int, little: Boolean): Int {
        val lo = read16(b, at, little)
        val hi = read16(b, at + 2, little)
        return if (little) lo or (hi shl 16) else (lo shl 16) or hi
    }
}
