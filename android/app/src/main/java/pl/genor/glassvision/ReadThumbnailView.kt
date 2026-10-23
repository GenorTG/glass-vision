package pl.genor.glassvision

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.ViewOutlineProvider
import androidx.appcompat.widget.AppCompatImageView
import androidx.core.content.ContextCompat

/** The Read text still (fit-center) with the recognized lines' [boxes] (normalized 0..1) outlined on it. */
class ReadThumbnailView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : AppCompatImageView(context, attrs) {
    private val dp = resources.displayMetrics.density
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.5f * dp
        color = ContextCompat.getColor(context, R.color.gv_primary)
    }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = ContextCompat.getColor(context, R.color.gv_primary) and 0x00FFFFFF or 0x33000000
    }
    private val image = RectF()
    private val box = RectF()

    var boxes: List<RectF> = emptyList()
        set(value) {
            field = value
            invalidate()
        }

    init {
        // Rounded like its background (tip_bg).
        outlineProvider = ViewOutlineProvider.BACKGROUND
        clipToOutline = true
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val d = drawable ?: return
        if (boxes.isEmpty() || d.intrinsicWidth <= 0 || d.intrinsicHeight <= 0) return
        image.set(0f, 0f, d.intrinsicWidth.toFloat(), d.intrinsicHeight.toFloat())
        imageMatrix.mapRect(image)
        image.offset(paddingLeft.toFloat(), paddingTop.toFloat())
        val radius = 3 * dp
        for (b in boxes) {
            box.set(
                image.left + b.left * image.width(),
                image.top + b.top * image.height(),
                image.left + b.right * image.width(),
                image.top + b.bottom * image.height(),
            )
            box.inset(-2 * dp, -2 * dp)
            canvas.drawRoundRect(box, radius, radius, fill)
            canvas.drawRoundRect(box, radius, radius, stroke)
        }
    }
}
