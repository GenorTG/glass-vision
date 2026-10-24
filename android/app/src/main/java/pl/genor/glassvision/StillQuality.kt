package pl.genor.glassvision

import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * Sharpness and brightness of the text in a picture, for the Read still's blur check.
 *
 * Sharpness = variance of the 4-neighbour Laplacian of the luma divided by the luma variance: the classic
 * focus/blur measure (https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/), normalized by contrast so a
 * re-metered (darker/brighter) still compares fairly. It still depends on the content and the scale, so it is only
 * ever compared on the same region at the same scale: the still's text region scaled to the live analysis frame's
 * resolution vs the sharpest live frame just before the shot ([BlurCheck]). Rotation doesn't change it.
 */
object Sharpness {
    /** [luma] row-major [width]×[height], 0..255; 0 for anything smaller than 3×3 or flat. */
    fun of(luma: IntArray, width: Int, height: Int): Float {
        if (width < 3 || height < 3) return 0f
        var sum = 0.0
        var sumSq = 0.0
        for (v in luma) {
            sum += v
            sumSq += v.toDouble() * v
        }
        val n = luma.size.toDouble()
        val variance = sumSq / n - (sum / n) * (sum / n)
        if (variance < 1.0) return 0f
        var lSum = 0.0
        var lSumSq = 0.0
        for (y in 1 until height - 1) {
            val row = y * width
            for (x in 1 until width - 1) {
                val i = row + x
                val l = luma[i - 1] + luma[i + 1] + luma[i - width] + luma[i + width] - 4 * luma[i]
                lSum += l
                lSumSq += l.toDouble() * l
            }
        }
        val m = ((width - 2) * (height - 2)).toDouble()
        val lVariance = lSumSq / m - (lSum / m) * (lSum / m)
        return (lVariance / variance).toFloat()
    }

    fun mean(luma: IntArray): Float = if (luma.isEmpty()) 0f else luma.sum().toFloat() / luma.size

    /** BT.601 luma of ARGB pixels (Bitmap.getPixels). */
    fun lumaOfArgb(pixels: IntArray): IntArray = IntArray(pixels.size) {
        val p = pixels[it]
        (77 * (p shr 16 and 0xFF) + 150 * (p shr 8 and 0xFF) + 29 * (p and 0xFF)) shr 8
    }

    /**
     * Luma of [rect] (pixels, clamped) in an RGBA_8888 plane (ImageAnalysis OUTPUT_IMAGE_FORMAT_RGBA_8888), every
     * [step]-th pixel in both directions (a cheap box-less downsample). Returns the luma and its size.
     */
    fun lumaOfRgba(
        buffer: ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        rect: PixelRect,
        step: Int = 1,
    ): Triple<IntArray, Int, Int> {
        val w = max(0, (rect.right - rect.left + step - 1) / step)
        val h = max(0, (rect.bottom - rect.top + step - 1) / step)
        val out = IntArray(w * h)
        for (j in 0 until h) {
            val row = (rect.top + j * step) * rowStride
            for (i in 0 until w) {
                val at = row + (rect.left + i * step) * pixelStride
                val r = buffer.get(at).toInt() and 0xFF
                val g = buffer.get(at + 1).toInt() and 0xFF
                val b = buffer.get(at + 2).toInt() and 0xFF
                out[j * w + i] = (77 * r + 150 * g + 29 * b) shr 8
            }
        }
        return Triple(out, w, h)
    }
}

/** Normalized rectangle 0..1 (x right, y down) and its 90°-step rotations, to move regions between frames. */
data class NormRect(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    /** This region after rotating its whole picture clockwise by [degrees] (a multiple of 90). */
    fun rotated(degrees: Int): NormRect = when (Math.floorMod(degrees, 360)) {
        90 -> NormRect(1f - bottom, left, 1f - top, right)
        180 -> NormRect(1f - right, 1f - bottom, 1f - left, 1f - top)
        270 -> NormRect(top, 1f - right, bottom, 1f - left)
        else -> this
    }

    /** Pixels in a [width]×[height] picture, clamped, at least 1×1 when the picture isn't empty. */
    fun toPixels(width: Int, height: Int): PixelRect {
        val l = (min(left, right).coerceIn(0f, 1f) * width).toInt().coerceIn(0, max(0, width - 1))
        val t = (min(top, bottom).coerceIn(0f, 1f) * height).toInt().coerceIn(0, max(0, height - 1))
        val r = (max(left, right).coerceIn(0f, 1f) * width).toInt().coerceIn(l + 1, max(l + 1, width))
        val b = (max(top, bottom).coerceIn(0f, 1f) * height).toInt().coerceIn(t + 1, max(t + 1, height))
        return PixelRect(l, t, r, b)
    }

    /** Grown by [margin] (fraction of this rect's size) on each side, clamped to the picture. */
    fun grown(margin: Float): NormRect {
        val dx = (right - left) * margin
        val dy = (bottom - top) * margin
        return NormRect((left - dx).coerceAtLeast(0f), (top - dy).coerceAtLeast(0f), (right + dx).coerceAtMost(1f), (bottom + dy).coerceAtMost(1f))
    }

    companion object {
        /** The middle of the picture: where the user aims when no text is marked. */
        val CENTRE = NormRect(0.3f, 0.35f, 0.7f, 0.65f)

        fun of(box: NormBox) = NormRect(min(box.x1, box.x2), min(box.y1, box.y2), max(box.x1, box.x2), max(box.y1, box.y2))
    }
}

/**
 * The text regions a Read still is about: the largest live text marks (the text pipeline's detections in the
 * analysis frame, upright portrait, normalized), or the centre. They aim the focus/metering and the blur check.
 */
object TextMarks {
    /** Up to [max] text marks of [result], largest first; marks too small to matter are skipped. */
    fun of(result: ServerMessage.Detections?, max: Int = 3): List<NormRect> {
        if (result == null) return emptyList()
        val marks = (result.tracks.ifEmpty { result.dets })
            // Held tracks too: OCR runs in a slow lane, so a mark on screen may be seconds old and still right.
            .filter { it.pipeline == TEXT_PIPELINE }
            .map { NormRect.of(it.box) }
            .filter { (it.right - it.left) * (it.bottom - it.top) >= MIN_AREA }
        return marks.sortedByDescending { (it.right - it.left) * (it.bottom - it.top) }.take(max)
    }

    /** Smallest rectangle around [regions] (null for none). */
    fun union(regions: List<NormRect>): NormRect? = regions.takeIf { it.isNotEmpty() }?.let { r ->
        NormRect(r.minOf { it.left }, r.minOf { it.top }, r.maxOf { it.right }, r.maxOf { it.bottom })
    }

    const val TEXT_PIPELINE = "text"
    private const val MIN_AREA = 0.0004f
}

/**
 * The still's sharpness against the sharpest live frame of the same region, both at the analysis frame's scale:
 * below [BLURRY_RATIO] of it the still is blurry (camera shake at the shutter, focus hunting) and is retaken once.
 * Calibrated on real sign photos with synthetic blur (sharp ≈ 1.7–2.6, Gaussian σ = 2 px at 4000 px ≈ 0.3–0.6,
 * 12 px motion ≈ 0.4–0.8): it only catches blur that costs OCR at the read scale. [measure] on phones.
 */
object BlurCheck {
    const val BLURRY_RATIO = 0.5f
    /** Mean luma (0..255) of the text region below which the picture is too dark to read well: suggest the torch. */
    const val DARK_LUMA = 50f

    /** Null when there is nothing to compare with (no live frame measured, flat region). */
    fun ratio(still: Float, reference: Float?): Float? =
        if (reference == null || reference <= 0f || still <= 0f) null else still / reference

    fun isBlurry(still: Float, reference: Float?): Boolean = ratio(still, reference)?.let { it < BLURRY_RATIO } == true
}
