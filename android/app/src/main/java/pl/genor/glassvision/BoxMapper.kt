package pl.genor.glassvision

import kotlin.math.max
import kotlin.math.min

/** Plain rect (JVM-testable, unlike android.graphics.RectF in local unit tests). */
data class ViewRect(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    val width: Float get() = right - left
    val height: Float get() = bottom - top
}

/**
 * Maps normalized boxes of the sent (upright) image onto a view showing it with FIT_CENTER.
 *
 * `mirrored` = the preview shows that image flipped horizontally (front camera: PreviewView mirrors it, the
 * frames we send are not), so x becomes `1 - x` and a box `[x1, x2]` lands at `[1 - x2, 1 - x1]`. Only
 * positions flip; the overlay still draws its labels readable.
 */
object BoxMapper {
    /** Where an image of [imageWidth]×[imageHeight] lands inside a [viewWidth]×[viewHeight] view under FIT_CENTER. */
    fun fitCenterRect(imageWidth: Int, imageHeight: Int, viewWidth: Int, viewHeight: Int): ViewRect {
        if (imageWidth <= 0 || imageHeight <= 0 || viewWidth <= 0 || viewHeight <= 0) {
            return ViewRect(0f, 0f, viewWidth.toFloat(), viewHeight.toFloat())
        }
        val scale = min(viewWidth.toFloat() / imageWidth, viewHeight.toFloat() / imageHeight)
        val w = imageWidth * scale
        val h = imageHeight * scale
        val left = (viewWidth - w) / 2f
        val top = (viewHeight - h) / 2f
        return ViewRect(left, top, left + w, top + h)
    }

    fun mapBox(box: NormBox, content: ViewRect, mirrored: Boolean = false): ViewRect {
        val a = mapX(box.x1, content, mirrored)
        val b = mapX(box.x2, content, mirrored)
        return ViewRect(left = min(a, b), top = mapY(box.y1, content), right = max(a, b), bottom = mapY(box.y2, content))
    }

    /** View x of a normalized x (box edge or keypoint), clamped into the content rect. */
    fun mapX(x: Float, content: ViewRect, mirrored: Boolean = false): Float {
        val n = x.coerceIn(0f, 1f)
        return content.left + (if (mirrored) 1f - n else n) * content.width
    }

    fun mapY(y: Float, content: ViewRect): Float = content.top + y.coerceIn(0f, 1f) * content.height
}
