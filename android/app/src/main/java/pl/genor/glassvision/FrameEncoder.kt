package pl.genor.glassvision

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import kotlin.math.max
import kotlin.math.roundToInt

/** Upright, scaled JPEG ready to send. [bytes] is valid for [length] bytes and reused by the next encode. */
class EncodedFrame(val bytes: ByteArray, val length: Int, val width: Int, val height: Int)

data class FrameSize(val width: Int, val height: Int)

/**
 * RGBA_8888 [ImageProxy] → upright bitmap (rotated by `rotationDegrees`) → downscaled so the
 * long side is at most `longSide` → JPEG. Not thread-safe: call from the single analysis thread.
 */
class FrameEncoder {
    private val jpegBuffer = ExposedByteArrayOutputStream(256 * 1024)

    fun encode(image: ImageProxy, longSide: Int, quality: Int): EncodedFrame {
        // Back sensors usually report 90°, front ones 270°: both end up upright (and unmirrored) here.
        val source = image.toBitmap()
        val upright = uprightBitmap(source, image.imageInfo.rotationDegrees, longSide)

        jpegBuffer.reset()
        upright.compress(Bitmap.CompressFormat.JPEG, quality, jpegBuffer)
        val frame = EncodedFrame(jpegBuffer.buffer(), jpegBuffer.size(), upright.width, upright.height)
        if (upright !== source) upright.recycle()
        source.recycle()
        return frame
    }

    companion object {
        /**
         * [source] rotated clockwise by [rotationDegrees] (CameraX `rotationDegrees` semantics) and scaled to
         * [uprightScaledSize] in one pass; scale is uniform so aspect is preserved. Returns [source] itself when
         * there is nothing to do, else a new bitmap (the caller recycles both).
         */
        fun uprightBitmap(source: Bitmap, rotationDegrees: Int, longSide: Int): Bitmap {
            val target = uprightScaledSize(source.width, source.height, rotationDegrees, longSide)
            val scale = max(target.width, target.height).toFloat() / max(source.width, source.height)
            if (rotationDegrees % 360 == 0 && scale == 1f) return source
            val matrix = Matrix().apply {
                postRotate(rotationDegrees.toFloat())
                postScale(scale, scale)
            }
            return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
        }

        /** Size after rotating by [rotationDegrees] and downscaling (never upscaling) to [longSide]. */
        fun uprightScaledSize(srcWidth: Int, srcHeight: Int, rotationDegrees: Int, longSide: Int): FrameSize {
            val swap = rotationDegrees % 180 != 0
            val w = if (swap) srcHeight else srcWidth
            val h = if (swap) srcWidth else srcHeight
            val scale = minOf(1f, longSide.toFloat() / max(w, h))
            return FrameSize((w * scale).roundToInt().coerceAtLeast(1), (h * scale).roundToInt().coerceAtLeast(1))
        }
    }

    private class ExposedByteArrayOutputStream(size: Int) : ByteArrayOutputStream(size) {
        fun buffer(): ByteArray = buf
    }
}
