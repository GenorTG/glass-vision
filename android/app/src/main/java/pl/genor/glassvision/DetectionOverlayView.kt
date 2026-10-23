package pl.genor.glassvision

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import android.text.TextPaint
import android.text.TextUtils
import android.util.AttributeSet
import android.util.TypedValue
import android.view.View
import kotlin.math.max
import kotlin.math.min

/**
 * Draws the latest detection result over a FIT_CENTER PreviewView. Boxes are normalized against the
 * upright image the client sent (`w`×`h` in the result), so they are mapped into the fit-center rect.
 * With [smoothing] it draws the server's tracks of all pipelines (held ones dimmed), gliding between results
 * via [BoxAnimator]; otherwise the raw per-frame detections. Keypoints are joined by the pipeline's skeleton
 * from `hello`. Results older than [STALE_MS] are not drawn. With [mirrored] (front camera) positions are
 * flipped horizontally to match the mirrored preview; labels stay readable. Main thread only.
 */
class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var result: ServerMessage.Detections? = null
    private var receivedAtUptime = 0L
    private val animator = BoxAnimator()
    private var pipelineIndex: Map<String, Int> = emptyMap()
    private var skeletons: Map<String, List<Pair<Int, Int>>> = emptyMap()

    var smoothing = true
        set(value) {
            field = value
            invalidate()
        }

    /** Time source for staleness and glide (snapshot tests pin it). */
    var clock: () -> Long = SystemClock::uptimeMillis

    /** The preview is mirrored (front camera): draw positions at `1 - x`. */
    var mirrored = false
        set(value) {
            field = value
            invalidate()
        }

    /** Box glide time constant in ms; 0 = boxes jump to each new result. */
    var glideMs: Int = BoxAnimator.DEFAULT_TAU_MS.toInt()
        set(value) {
            field = value
            animator.tauMs = value.toFloat()
        }

    private val density = resources.displayMetrics.density
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = BOX_STROKE_DP * density
    }
    private val bonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f * density
        strokeCap = Paint.Cap.ROUND
    }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val labelBgPaint = Paint().apply { style = Paint.Style.FILL }
    private val textPaint = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, 13f, resources.displayMetrics)
    }
    private val textPad = 3f * density
    private val minLabelWidth = 150f * density
    private val dotRadius = 3f * density
    private val hsv = FloatArray(3)
    private val clearRunnable = Runnable { invalidate() }

    /** Size of the frame the last result refers to (its aspect = the camera image's); null without one. */
    val frameSize: Pair<Int, Int>? get() = result?.let { it.width to it.height }

    /** Pipelines from `hello`: their order fixes the colours, their skeletons join keypoints. */
    fun setPipelines(specs: List<PipelineSpec>) {
        pipelineIndex = specs.withIndex().associate { (i, s) -> s.id to i }
        skeletons = specs.filter { it.skeleton.isNotEmpty() }.associate { it.id to it.skeleton }
        invalidate()
    }

    fun show(detections: ServerMessage.Detections) {
        result = detections
        receivedAtUptime = clock()
        animator.setTargets(detections.tracks, receivedAtUptime)
        removeCallbacks(clearRunnable)
        // Re-draw once the result goes stale so boxes vanish even if no new result arrives.
        postDelayed(clearRunnable, STALE_MS + 16)
        invalidate()
    }

    fun clear() {
        result = null
        animator.clear()
        removeCallbacks(clearRunnable)
        invalidate()
    }

    override fun onDetachedFromWindow() {
        removeCallbacks(clearRunnable)
        super.onDetachedFromWindow()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val r = result ?: return
        val now = clock()
        if (now - receivedAtUptime > STALE_MS) return

        val content = BoxMapper.fitCenterRect(r.width, r.height, width, height)
        val toDraw = if (smoothing) {
            animator.epsilon = ARRIVED_PX / max(1f, max(content.width, content.height))
            // Only keep the frame loop alive while a box still glides: static scenes cost no frames.
            if (animator.step(now)) postInvalidateOnAnimation()
            animator.current()
        } else {
            r.dets
        }
        for (det in toDraw) drawDetection(canvas, det, content)
    }

    private fun drawDetection(canvas: Canvas, det: Detection, content: ViewRect) {
        // Held tracks (missed this frame, shown at the predicted position) are drawn dimmer.
        val alpha = if (det.seenMsAgo > 0) HELD_ALPHA else 0xFF
        hsv[0] = TrackStyle.hue(det.pipeline, pipelineIndex[det.pipeline] ?: -1, det.label)
        hsv[1] = 0.85f
        hsv[2] = 1f
        val color = Color.HSVToColor(hsv)
        val b = BoxMapper.mapBox(det.box, content, mirrored)

        boxPaint.color = color
        boxPaint.alpha = alpha
        boxPaint.strokeWidth = (if (TrackStyle.isThin(det)) THIN_STROKE_DP else BOX_STROKE_DP) * density
        canvas.drawRect(b.left, b.top, b.right, b.bottom, boxPaint)

        if (det.points.isNotEmpty()) drawPoints(canvas, det, content, color, alpha)
        if (TrackStyle.hasLabel(det)) drawLabel(canvas, det, b, content, color, alpha)
    }

    private fun drawPoints(canvas: Canvas, det: Detection, content: ViewRect, color: Int, alpha: Int) {
        val pts = det.points
        val px = { i: Int -> BoxMapper.mapX(pts[i].x, content, mirrored) }
        val py = { i: Int -> BoxMapper.mapY(pts[i].y, content) }
        bonePaint.color = color
        bonePaint.alpha = alpha
        for ((a, c) in skeletons[det.pipeline].orEmpty()) {
            if (a >= pts.size || c >= pts.size) continue
            if (pts[a].v < MIN_VISIBILITY || pts[c].v < MIN_VISIBILITY) continue
            canvas.drawLine(px(a), py(a), px(c), py(c), bonePaint)
        }
        dotPaint.color = Color.WHITE
        dotPaint.alpha = alpha
        for (i in pts.indices) {
            if (pts[i].v < MIN_VISIBILITY) continue
            canvas.drawCircle(px(i), py(i), dotRadius, dotPaint)
        }
    }

    private fun drawLabel(canvas: Canvas, det: Detection, b: ViewRect, content: ViewRect, color: Int, alpha: Int) {
        val fm = textPaint.fontMetrics
        val textHeight = fm.descent - fm.ascent
        // Long labels (OCR text) are clamped to the box width (at least minLabelWidth) and ellipsized.
        val maxWidth = min(max(b.width, minLabelWidth), content.width * MAX_LABEL_FRACTION) - 2 * textPad
        val text = TextUtils.ellipsize(TrackStyle.labelText(det), textPaint, maxWidth, TextUtils.TruncateAt.END).toString()
        val textWidth = textPaint.measureText(text)
        // Label above the box, or inside it when the box touches the top edge.
        val labelTop = if (b.top - textHeight - 2 * textPad >= content.top) b.top - textHeight - 2 * textPad else b.top
        labelBgPaint.color = (color and 0x00FFFFFF) or ((0xCC * alpha / 0xFF) shl 24)
        textPaint.alpha = alpha
        canvas.drawRect(b.left, labelTop, b.left + textWidth + 2 * textPad, labelTop + textHeight + 2 * textPad, labelBgPaint)
        canvas.drawText(text, b.left + textPad, labelTop + textPad - fm.ascent, textPaint)
    }

    companion object {
        const val STALE_MS = 1000L
        const val HELD_ALPHA = 0x99
        const val MIN_VISIBILITY = 0.3f
        private const val BOX_STROKE_DP = 2.5f
        private const val THIN_STROKE_DP = 1.25f
        private const val MAX_LABEL_FRACTION = 0.7f
        /** A gliding box within this many pixels of its target counts as arrived. */
        private const val ARRIVED_PX = 0.5f
    }
}
