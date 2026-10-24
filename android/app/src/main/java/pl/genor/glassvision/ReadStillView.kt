package pl.genor.glassvision

import android.content.Context
import android.graphics.Canvas
import android.graphics.CornerPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.drawable.BitmapDrawable
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.ViewOutlineProvider
import androidx.appcompat.widget.AppCompatImageView
import androidx.core.content.ContextCompat
import kotlin.math.max

/**
 * The Read text still (fit-center) with its [blocks]: numbered outlines (the numbers match the cards), or with
 * [overlay] set, each block's background sampled from the picture, filled in, and its text drawn inside at the largest
 * size that fits ([OverlayLayout]), rotated with the block. Tapping a block calls [onBlockTap]; [selected] is
 * highlighted.
 */
class ReadStillView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : AppCompatImageView(context, attrs) {
    private val dp = resources.displayMetrics.density
    private val primary = ContextCompat.getColor(context, R.color.gv_primary)
    private val suspectColor = ContextCompat.getColor(context, R.color.gv_status_busy)
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1.5f * dp
        strokeJoin = Paint.Join.ROUND
        pathEffect = CornerPathEffect(3 * dp)
    }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        pathEffect = CornerPathEffect(3 * dp)
    }
    private val erase = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL_AND_STROKE
        strokeJoin = Paint.Join.ROUND
    }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { isSubpixelText = true }
    private val badge = Paint(Paint.ANTI_ALIAS_FLAG)
    private val badgeText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.gv_on_primary)
        textSize = 11 * dp
        textAlign = Paint.Align.CENTER
        isFakeBoldText = true
    }
    private val image = RectF()
    private val path = Path()

    /** The blocks to mark, coordinates normalized against the picture. */
    var blocks: List<ReadBlock> = emptyList()
        set(value) {
            field = value
            colors = null
            invalidate()
        }

    /** Text painted over each block (aligned with [blocks]; a null entry leaves that block as it is), or null = outlines. */
    var overlay: List<String?>? = null
        set(value) {
            field = value
            invalidate()
        }

    /** Index into [blocks] shown highlighted, or -1. */
    var selected: Int = -1
        set(value) {
            field = value
            invalidate()
        }

    var onBlockTap: ((Int) -> Unit)? = null

    /** The height follows the picture's aspect ratio (up to [maxFitHeight]) instead of the layout height. */
    var fitToImage: Boolean = false
        set(value) {
            if (field == value) return
            field = value
            requestLayout()
        }

    var maxFitHeight: Int = (520 * dp).toInt()

    /** Background and text colour per block, sampled from the picture on first use. */
    private var colors: List<Pair<Int, Int>>? = null

    init {
        // Rounded like its background (tip_bg).
        outlineProvider = ViewOutlineProvider.BACKGROUND
        clipToOutline = true
    }

    override fun setImageBitmap(bm: android.graphics.Bitmap?) {
        super.setImageBitmap(bm)
        colors = null
        requestLayout()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val d = drawable
        if (fitToImage && d != null && d.intrinsicWidth > 0 && d.intrinsicHeight > 0) {
            val w = MeasureSpec.getSize(widthMeasureSpec)
            setMeasuredDimension(w, (w.toLong() * d.intrinsicHeight / d.intrinsicWidth).toInt().coerceAtMost(maxFitHeight))
            return
        }
        super.onMeasure(widthMeasureSpec, heightMeasureSpec)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (blocks.isEmpty() || !imageRect()) return
        val quads = blocks.map(::viewQuad)
        val painted = overlay
        if (painted != null) {
            val sampled = colors ?: sample().also { colors = it }
            for (i in blocks.indices) painted.getOrNull(i)?.let { paint(canvas, quads[i], it, sampled[i], blocks[i]) }
        }
        for ((i, q) in quads.withIndex()) {
            val translated = painted?.getOrNull(i) != null
            if (translated && i != selected) continue
            val color = if (blocks[i].suspect) suspectColor else primary
            quadPath(q, 2 * dp)
            if (!translated) {
                fill.color = color and 0x00FFFFFF or (if (i == selected) 0x55000000 else 0x33000000)
                canvas.drawPath(path, fill)
            }
            stroke.color = color
            stroke.strokeWidth = (if (i == selected) 3f else 1.5f) * dp
            canvas.drawPath(path, stroke)
        }
        if (painted == null) for ((i, q) in quads.withIndex()) number(canvas, q, i)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val tap = onBlockTap ?: return super.onTouchEvent(event)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> return true
            MotionEvent.ACTION_UP -> {
                if (imageRect()) {
                    val quads = blocks.map(::viewQuad)
                    var hit = OverlayLayout.hit(quads, event.x, event.y)
                    if (hit < 0) {
                        // Small blocks: the nearest one within a finger's reach.
                        val reach = 16 * dp
                        hit = quads.indices.filter { i ->
                            val b = quads[i].bounds()
                            event.x in b.left - reach..b.right + reach && event.y in b.top - reach..b.bottom + reach
                        }.minByOrNull { i -> quads[i].bounds().let { b -> (b.width * b.height) } } ?: -1
                    }
                    if (hit >= 0) tap(hit)
                }
                performClick()
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    override fun performClick(): Boolean = super.performClick()

    /** [image] = where the picture is drawn in this view; false without a picture. */
    private fun imageRect(): Boolean {
        val d = drawable ?: return false
        if (d.intrinsicWidth <= 0 || d.intrinsicHeight <= 0) return false
        image.set(0f, 0f, d.intrinsicWidth.toFloat(), d.intrinsicHeight.toFloat())
        imageMatrix.mapRect(image)
        image.offset(paddingLeft.toFloat(), paddingTop.toFloat())
        return true
    }

    private fun viewQuad(b: ReadBlock): Quad = b.corners.map { Pt(image.left + it.x * image.width(), image.top + it.y * image.height()) }

    /** [q] grown by [grow] px on every side (around its centre) into [path]. */
    private fun quadPath(q: Quad, grow: Float) {
        val cx = q.points.sumOf { it.x.toDouble() }.toFloat() / 4f
        val cy = q.points.sumOf { it.y.toDouble() }.toFloat() / 4f
        path.reset()
        for ((i, p) in q.points.withIndex()) {
            val dx = p.x - cx
            val dy = p.y - cy
            val len = max(1f, kotlin.math.hypot(dx, dy))
            val x = p.x + dx / len * grow * 1.41f
            val y = p.y + dy / len * grow * 1.41f
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        path.close()
    }

    private fun number(canvas: Canvas, q: Quad, i: Int) {
        val r = 9 * dp
        val p = q.points[0]
        // Left of the block's first line, so it doesn't cover the text it numbers.
        val x = (p.x - r - 3 * dp).coerceIn(image.left + r, image.right - r)
        val y = (p.y + r * 0.5f).coerceIn(image.top + r, image.bottom - r)
        badge.color = if (blocks[i].suspect) suspectColor else primary
        canvas.drawCircle(x, y, r, badge)
        badgeText.color = ContextCompat.getColor(context, if (blocks[i].suspect) R.color.gv_on_status_busy else R.color.gv_on_primary)
        canvas.drawText((i + 1).toString(), x, y - (badgeText.descent() + badgeText.ascent()) / 2f, badgeText)
    }

    /** Erase the block with its background colour, then [label] inside it, rotated with the block. */
    private fun paint(canvas: Canvas, q: Quad, label: String, colors: Pair<Int, Int>, block: ReadBlock) {
        val thick = q.thickness
        val len = q.length
        if (thick < 2f || len < 2f) return
        val originalLine = thick / max(1, block.lines.size)
        // Grow by a share of one line, not of the block: the sign's plate often ends just past its text.
        val grow = 0.22f * originalLine
        erase.color = colors.first
        erase.strokeWidth = 2 * grow
        quadPath(q, 0f)
        canvas.drawPath(path, erase)

        val p0 = q.points[0]
        canvas.save()
        canvas.translate(p0.x, p0.y)
        canvas.rotate(q.angle)
        canvas.clipRect(-grow, -grow, len + grow, thick + grow)
        val pad = max(dp, 0.04f * thick)
        val w = len - 2 * pad + grow
        val h = thick - 2 * pad + grow
        text.textSize = MEASURE_SIZE
        val lineHeight = text.fontSpacing / MEASURE_SIZE
        val fit = OverlayLayout.fit(
            label, w, h, { text.measureText(it) / MEASURE_SIZE }, lineHeight,
            minSize = MIN_TEXT_SP * dp, maxSize = max(MIN_TEXT_SP * dp, 1.1f * originalLine),
        )
        text.textSize = fit.size
        text.color = colors.second
        val step = text.fontSpacing
        val top = max(-grow / 2, (thick - fit.lines.size * step) / 2f)
        val rtl = TextBlocks.isRtl(label)
        for ((i, line) in fit.lines.withIndex()) {
            val lw = text.measureText(line)
            val x = when {
                rtl -> len - pad - lw
                block.kind == BlockKind.LIST -> (len - lw) / 2f
                else -> pad - grow / 2
            }
            canvas.drawText(line, x, top + i * step - text.ascent(), text)
        }
        canvas.restore()
    }

    /** Per block: (background = median of a ring around it, text colour from its glyphs — [OverlayLayout.textColor]). */
    private fun sample(): List<Pair<Int, Int>> {
        val fallback = OverlayLayout.BLACK to OverlayLayout.WHITE
        val bitmap = (drawable as? BitmapDrawable)?.bitmap ?: return blocks.map { fallback }
        val bw = bitmap.width
        val bh = bitmap.height
        fun px(x: Float, y: Float): Int = bitmap.getPixel(x.toInt().coerceIn(0, bw - 1), y.toInt().coerceIn(0, bh - 1))
        return blocks.map { block ->
            val q = block.corners.map { Pt(it.x * bw, it.y * bh) }
            val b = q.bounds()
            // Just outside the text: a share of one line (a block-sized margin would land beyond a small sign's plate).
            val pad = max(2f, 0.2f * q.thickness / max(1, block.lines.size))
            val ring = ArrayList<Int>()
            for (k in 0..RING_STEPS) {
                val t = k / RING_STEPS.toFloat()
                val x = b.left - pad + t * (b.width + 2 * pad)
                val y = b.top - pad + t * (b.height + 2 * pad)
                ring += px(x, b.top - pad)
                ring += px(x, b.bottom + pad)
                ring += px(b.left - pad, y)
                ring += px(b.right + pad, y)
            }
            val inside = ArrayList<Int>()
            for (gy in 0 until INSIDE_ROWS) for (gx in 0 until INSIDE_COLS) {
                val x = b.left + (gx + 0.5f) * b.width / INSIDE_COLS
                val y = b.top + (gy + 0.5f) * b.height / INSIDE_ROWS
                if (OverlayLayout.contains(q.points, x, y)) inside += px(x, y)
            }
            val bg = OverlayLayout.median(ring) ?: return@map fallback
            bg to OverlayLayout.textColor(bg, OverlayLayout.glyphColor(inside, bg))
        }
    }

    private companion object {
        const val MEASURE_SIZE = 100f
        const val MIN_TEXT_SP = 7f
        const val RING_STEPS = 16
        const val INSIDE_COLS = 24
        const val INSIDE_ROWS = 10
    }
}
