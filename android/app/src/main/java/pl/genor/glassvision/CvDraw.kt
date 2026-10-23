package pl.genor.glassvision

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * An RGB image as packed `0xRRGGBB` ints, row-major. Pure Kotlin, so the OpenPose renderer runs (and is tested)
 * on the JVM; the phone turns it into a Bitmap only to encode PNG/JPEG.
 */
class RgbRaster(val width: Int, val height: Int, val pixels: IntArray = IntArray(width * height)) {
    init {
        require(width > 0 && height > 0 && pixels.size == width * height) { "bad raster ${width}x$height" }
    }

    operator fun get(x: Int, y: Int): Int = pixels[y * width + x]

    fun copy(): RgbRaster = RgbRaster(width, height, pixels.copyOf())

    companion object {
        fun rgb(r: Int, g: Int, b: Int): Int = (r and 0xFF shl 16) or (g and 0xFF shl 8) or (b and 0xFF)
    }
}

/**
 * The few OpenCV drawing primitives the OpenPose renderer uses, ported line by line from OpenCV 5.0.0
 * `modules/imgproc/src/drawing.cpp` (Apache-2.0; the cv2 the server runs), so renders match the server's pixel for
 * pixel: `cv2.ellipse2Poly`, `cv2.fillConvexPoly` (LINE_8, any shift), `cv2.circle` filled (LINE_8) and `cv2.line`
 * (LINE_8, any thickness). Only the paths those calls take are ported (no anti-aliasing, 3-channel images).
 * https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/drawing.cpp
 */
object CvDraw {
    private const val XY_SHIFT = 16
    private const val XY_ONE = 1L shl XY_SHIFT

    data class Pt(val x: Long, val y: Long)

    /** C's cvRound: round half to even (lrint under the default rounding mode). */
    fun cvRound(v: Double): Long = Math.rint(v).toLong()

    /** `saturate_cast<uchar>(double)`: what cv2 makes of a float colour channel. */
    fun channel(v: Double): Int = cvRound(v).coerceIn(0L, 255L).toInt()

    // --- ellipse2Poly ----------------------------------------------------------------------------

    fun ellipse2Poly(cx: Int, cy: Int, axisW: Int, axisH: Int, angleIn: Int, arcStartIn: Int, arcEndIn: Int, delta: Int): List<Pt> {
        require(delta in 1..180)
        var angle = angleIn
        var arcStart = arcStartIn
        var arcEnd = arcEndIn
        while (angle < 0) angle += 360
        while (angle > 360) angle -= 360
        if (arcStart > arcEnd) {
            val i = arcStart
            arcStart = arcEnd
            arcEnd = i
        }
        while (arcStart < 0) {
            arcStart += 360
            arcEnd += 360
        }
        while (arcEnd > 360) {
            arcEnd -= 360
            arcStart -= 360
        }
        if (arcEnd - arcStart > 360) {
            arcStart = 0
            arcEnd = 360
        }
        // sincos(angle, alpha, beta): alpha = cos, beta = sin, both floats from the table.
        val a = angle + if (angle < 0) 360 else 0
        val beta = SIN_TABLE[a]
        val alpha = SIN_TABLE[450 - a]

        val out = ArrayList<Pt>()
        var prev: Pt? = null
        var i = arcStart
        while (i < arcEnd + delta) {
            var ang = i
            if (ang > arcEnd) ang = arcEnd
            if (ang < 0) ang += 360
            val x = axisW * SIN_TABLE[450 - ang].toDouble()
            val y = axisH * SIN_TABLE[ang].toDouble()
            val px = cx + x * alpha - y * beta
            val py = cy + x * beta + y * alpha
            val pt = Pt(cvRound(px), cvRound(py))
            if (pt != prev) {
                out += pt
                prev = pt
            }
            i += delta
        }
        if (out.size == 1) return listOf(Pt(cx.toLong(), cy.toLong()), Pt(cx.toLong(), cy.toLong()))
        return out
    }

    // --- lines -----------------------------------------------------------------------------------

    /** clipLine(Size2l, pt1, pt2): the points as OpenCV leaves them, and its result (both ends inside). */
    private class Clip(val p1: Pt, val p2: Pt, val inside: Boolean)

    private fun clipLine(w: Long, h: Long, p1: Pt, p2: Pt): Clip {
        val right = w - 1
        val bottom = h - 1
        if (w <= 0 || h <= 0) return Clip(p1, p2, false)
        var x1 = p1.x
        var y1 = p1.y
        var x2 = p2.x
        var y2 = p2.y
        fun code(x: Long, y: Long) =
            (if (x < 0) 1 else 0) + (if (x > right) 2 else 0) + (if (y < 0) 4 else 0) + (if (y > bottom) 8 else 0)
        var c1 = code(x1, y1)
        var c2 = code(x2, y2)
        if ((c1 and c2) == 0 && (c1 or c2) != 0) {
            var a: Long
            if (c1 and 12 != 0) {
                a = if (c1 < 8) 0 else bottom
                x1 += ((a - y1).toDouble() * (x2 - x1) / (y2 - y1)).toLong()
                y1 = a
                c1 = (if (x1 < 0) 1 else 0) + (if (x1 > right) 2 else 0)
            }
            if (c2 and 12 != 0) {
                a = if (c2 < 8) 0 else bottom
                x2 += ((a - y2).toDouble() * (x2 - x1) / (y2 - y1)).toLong()
                y2 = a
                c2 = (if (x2 < 0) 1 else 0) + (if (x2 > right) 2 else 0)
            }
            if ((c1 and c2) == 0 && (c1 or c2) != 0) {
                if (c1 != 0) {
                    a = if (c1 == 1) 0 else right
                    y1 += ((a - x1).toDouble() * (y2 - y1) / (x2 - x1)).toLong()
                    x1 = a
                    c1 = 0
                }
                if (c2 != 0) {
                    a = if (c2 == 1) 0 else right
                    y2 += ((a - x2).toDouble() * (y2 - y1) / (x2 - x1)).toLong()
                    x2 = a
                    c2 = 0
                }
            }
        }
        return Clip(Pt(x1, y1), Pt(x2, y2), (c1 or c2) == 0)
    }

    /** Line(img, pt1, pt2, color, 8): LineIterator, 8-connected, left to right. */
    private fun line8(img: RgbRaster, pt1In: Pt, pt2In: Pt, color: Int) {
        var pt1 = pt1In
        var pt2 = pt2In
        if (pt1.x !in 0 until img.width || pt2.x !in 0 until img.width ||
            pt1.y !in 0 until img.height || pt2.y !in 0 until img.height
        ) {
            val clipped = clipLine(img.width.toLong(), img.height.toLong(), pt1, pt2)
            if (!clipped.inside) return
            pt1 = clipped.p1
            pt2 = clipped.p2
        }
        var deltaX = 1
        var deltaY = 1
        var dx = (pt2.x - pt1.x).toInt()
        var dy = (pt2.y - pt1.y).toInt()
        if (dx < 0) { // leftToRight
            dx = -dx
            dy = -dy
            pt1 = pt2
        }
        if (dy < 0) {
            dy = -dy
            deltaY = -1
        }
        val vert = dy > dx
        if (vert) {
            val t = dx
            dx = dy
            dy = t
            val td = deltaX
            deltaX = deltaY
            deltaY = td
        }
        var err = dx - (dy + dy)
        val plusDelta = dx + dx
        val minusDelta = -(dy + dy)
        var minusShift = deltaX
        var plusShift = 0
        var minusStep = 0
        var plusStep = deltaY
        val count = dx + 1
        if (vert) {
            var t = plusStep
            plusStep = plusShift
            plusShift = t
            t = minusStep
            minusStep = minusShift
            minusShift = t
        }
        var x = pt1.x.toInt()
        var y = pt1.y.toInt()
        for (i in 0 until count) {
            img.pixels[y * img.width + x] = color
            val mask = if (err < 0) -1 else 0
            err += minusDelta + (plusDelta and mask)
            x += minusShift + (plusShift and mask)
            y += minusStep + (plusStep and mask)
        }
    }

    /** Line2(img, pt1, pt2, color): fixed-point (XY_SHIFT) line used by FillConvexPoly with shift > 0. */
    private fun line2(img: RgbRaster, pt1In: Pt, pt2In: Pt, color: Int) {
        val clipped = clipLine(img.width.toLong() shl XY_SHIFT, img.height.toLong() shl XY_SHIFT, pt1In, pt2In)
        if (!clipped.inside) return
        var p1x = clipped.p1.x
        var p1y = clipped.p1.y
        var p2x = clipped.p2.x
        var p2y = clipped.p2.y
        var dx = p2x - p1x
        var dy = p2y - p1y
        val j = if (dx < 0) -1L else 0L
        val ax = (dx xor j) - j
        val i = if (dy < 0) -1L else 0L
        val ay = (dy xor i) - i
        val xStep: Long
        val yStep: Long
        var ecount: Int
        if (ax > ay) {
            dy = (dy xor j) - j
            p1x = p1x xor (p2x and j); p2x = p2x xor (p1x and j); p1x = p1x xor (p2x and j)
            p1y = p1y xor (p2y and j); p2y = p2y xor (p1y and j); p1y = p1y xor (p2y and j)
            xStep = XY_ONE
            yStep = dy * (1L shl XY_SHIFT) / (ax or 1)
            ecount = ((p2x - p1x) shr XY_SHIFT).toInt()
        } else {
            dx = (dx xor i) - i
            p1x = p1x xor (p2x and i); p2x = p2x xor (p1x and i); p1x = p1x xor (p2x and i)
            p1y = p1y xor (p2y and i); p2y = p2y xor (p1y and i); p1y = p1y xor (p2y and i)
            xStep = dx * (1L shl XY_SHIFT) / (ay or 1)
            yStep = XY_ONE
            ecount = ((p2y - p1y) shr XY_SHIFT).toInt()
        }
        p1x += XY_ONE shr 1
        p1y += XY_ONE shr 1
        put(img, ((p2x + (XY_ONE shr 1)) shr XY_SHIFT).toInt(), ((p2y + (XY_ONE shr 1)) shr XY_SHIFT).toInt(), color)
        if (ax > ay) {
            p1x = p1x shr XY_SHIFT
            while (ecount >= 0) {
                put(img, p1x.toInt(), (p1y shr XY_SHIFT).toInt(), color)
                p1x++
                p1y += yStep
                ecount--
            }
        } else {
            p1y = p1y shr XY_SHIFT
            while (ecount >= 0) {
                put(img, (p1x shr XY_SHIFT).toInt(), p1y.toInt(), color)
                p1x += xStep
                p1y++
                ecount--
            }
        }
    }

    private fun put(img: RgbRaster, x: Int, y: Int, color: Int) {
        if (x in 0 until img.width && y in 0 until img.height) img.pixels[y * img.width + x] = color
    }

    /** ICV_HLINE: x1..x2 inclusive on row y (callers clip). */
    private fun hline(img: RgbRaster, y: Int, x1: Long, x2: Long, color: Int) {
        val row = y * img.width
        var x = x1
        while (x <= x2) {
            img.pixels[row + x.toInt()] = color
            x++
        }
    }

    // --- FillConvexPoly --------------------------------------------------------------------------

    /** `cv2.fillConvexPoly(img, pts, color)` with LINE_8 and [shift] fractional bits in [v]. */
    fun fillConvexPoly(img: RgbRaster, v: List<Pt>, color: Int, shift: Int = 0) {
        val npts = v.size
        if (npts <= 0) return
        val delta = (1 shl shift) shr 1
        var imin = 0
        var edges = npts
        val delta1 = XY_ONE shr 1 // line_type < LINE_AA
        val delta2 = XY_ONE shr 1
        var p0 = Pt(v[npts - 1].x shl (XY_SHIFT - shift), v[npts - 1].y shl (XY_SHIFT - shift))
        var xmin = v[0].x
        var xmax = v[0].x
        var ymin = v[0].y
        var ymax = v[0].y
        for (i in 0 until npts) {
            val p0r = v[i]
            if (p0r.y < ymin) {
                ymin = p0r.y
                imin = i
            }
            ymax = max(ymax, p0r.y)
            xmax = max(xmax, p0r.x)
            xmin = min(xmin, p0r.x)
            val p = Pt(p0r.x shl (XY_SHIFT - shift), p0r.y shl (XY_SHIFT - shift))
            if (shift == 0) {
                line8(img, Pt(p0.x shr XY_SHIFT, p0.y shr XY_SHIFT), Pt(p.x shr XY_SHIFT, p.y shr XY_SHIFT), color)
            } else {
                line2(img, p0, p, color)
            }
            p0 = p
        }
        xmin = (xmin + delta) shr shift
        xmax = (xmax + delta) shr shift
        ymin = (ymin + delta) shr shift
        ymax = (ymax + delta) shr shift
        if (npts < 3 || xmax.toInt() < 0 || ymax.toInt() < 0 || xmin.toInt() >= img.width || ymin.toInt() >= img.height) return
        ymax = min(ymax, (img.height - 1).toLong())

        val eIdx = intArrayOf(imin, imin)
        val eDi = intArrayOf(1, npts - 1)
        val eX = longArrayOf(-XY_ONE, -XY_ONE)
        val eDx = longArrayOf(0, 0)
        var y = ymin.toInt()
        val eYe = intArrayOf(y, y)
        do {
            for (i in 0..1) {
                if (y >= eYe[i]) {
                    var idx0 = eIdx[i]
                    val di = eDi[i]
                    var idx = idx0 + di
                    if (idx >= npts) idx -= npts
                    var ty: Int
                    while (edges-- > 0) {
                        ty = ((v[idx].y + delta) shr shift).toInt()
                        if (ty > y) {
                            var xs = v[idx0].x
                            var xe = v[idx].x
                            if (shift != XY_SHIFT) {
                                xs = xs shl (XY_SHIFT - shift)
                                xe = xe shl (XY_SHIFT - shift)
                            }
                            eYe[i] = ty
                            eDx[i] = ((xe - xs) * 2 + (ty.toLong() - y)) / (2 * (ty.toLong() - y))
                            eX[i] = xs
                            eIdx[i] = idx
                            break
                        }
                        idx0 = idx
                        idx += di
                        if (idx >= npts) idx -= npts
                    }
                }
            }
            if (edges < 0) break
            if (y >= 0) {
                var left = 0
                var right = 1
                if (eX[0] > eX[1]) {
                    left = 1
                    right = 0
                }
                var xx1 = ((eX[left] + delta1) shr XY_SHIFT).toInt()
                var xx2 = ((eX[right] + delta2) shr XY_SHIFT).toInt()
                if (xx2 >= 0 && xx1 < img.width) {
                    if (xx1 < 0) xx1 = 0
                    if (xx2 >= img.width) xx2 = img.width - 1
                    hline(img, y, xx1.toLong(), xx2.toLong(), color)
                }
            }
            eX[0] += eDx[0]
            eX[1] += eDx[1]
        } while (++y <= ymax.toInt())
    }

    // --- circle (filled, LINE_8) -----------------------------------------------------------------

    /** `cv2.circle(img, center, radius, color, thickness=-1)`: OpenCV's `Circle(..., fill=1)`. */
    fun fillCircle(img: RgbRaster, cx: Int, cy: Int, radius: Int, color: Int) {
        val w = img.width
        val h = img.height
        var err = 0L
        var dx = radius.toLong()
        var dy = 0L
        var plus = 1L
        var minus = (radius.toLong() shl 1) - 1
        val inside = cx >= radius && cx < w - radius && cy >= radius && cy < h - radius
        while (dx >= dy) {
            val y11 = cy - dy
            val y12 = cy + dy
            val y21 = cy - dx
            val y22 = cy + dx
            var x11 = cx - dx
            var x12 = cx + dx
            var x21 = cx - dy
            var x22 = cx + dy
            if (inside) {
                hline(img, y11.toInt(), x11, x12, color)
                hline(img, y12.toInt(), x11, x12, color)
                hline(img, y21.toInt(), x21, x22, color)
                hline(img, y22.toInt(), x21, x22, color)
            } else if (x11 < w && x12 >= 0 && y21 < h && y22 >= 0) {
                x11 = max(x11, 0L)
                x12 = min(x12, (w - 1).toLong())
                if (y11 in 0 until h) hline(img, y11.toInt(), x11, x12, color)
                if (y12 in 0 until h) hline(img, y12.toInt(), x11, x12, color)
                if (x21 < w && x22 >= 0) {
                    x21 = max(x21, 0L)
                    x22 = min(x22, (w - 1).toLong())
                    if (y21 in 0 until h) hline(img, y21.toInt(), x21, x22, color)
                    if (y22 in 0 until h) hline(img, y22.toInt(), x21, x22, color)
                }
            }
            dy++
            err += plus
            plus += 2
            val mask = (if (err <= 0) 1L else 0L) - 1
            err -= minus and mask
            dx += mask
            minus -= mask and 2
        }
    }

    // --- line with thickness (LINE_8) ------------------------------------------------------------

    /** `cv2.line(img, p0, p1, color, thickness)` with LINE_8 and shift 0: ThickLine with round caps (flags 3). */
    fun line(img: RgbRaster, x0: Int, y0: Int, x1: Int, y1: Int, color: Int, thicknessIn: Int) {
        var p0 = Pt(x0.toLong(), y0.toLong())
        var p1 = Pt(x1.toLong(), y1.toLong())
        var thickness = thicknessIn
        val inBounds = { p: Pt -> p.x >= 0 && p.y >= 0 && p.x < img.width && p.y < img.height }
        if (thickness > 1 && (!inBounds(p0) || !inBounds(p1))) {
            val m = thickness.toLong()
            // OpenCV ignores clipLine's result here and keeps whatever it wrote back.
            val clipped = clipLine(img.width + 2 * m, img.height + 2 * m, Pt(p0.x + m, p0.y + m), Pt(p1.x + m, p1.y + m))
            p0 = Pt(clipped.p1.x - m, clipped.p1.y - m)
            p1 = Pt(clipped.p2.x - m, clipped.p2.y - m)
        }
        p0 = Pt(p0.x shl XY_SHIFT, p0.y shl XY_SHIFT)
        p1 = Pt(p1.x shl XY_SHIFT, p1.y shl XY_SHIFT)
        if (thickness <= 1) {
            line8(
                img,
                Pt((p0.x + (XY_ONE shr 1)) shr XY_SHIFT, (p0.y + (XY_ONE shr 1)) shr XY_SHIFT),
                Pt((p1.x + (XY_ONE shr 1)) shr XY_SHIFT, (p1.y + (XY_ONE shr 1)) shr XY_SHIFT),
                color,
            )
            return
        }
        val invOne = 1.0 / XY_ONE.toDouble()
        val dx = (p0.x - p1.x) * invOne
        val dy = (p1.y - p0.y) * invOne
        var r = dx * dx + dy * dy
        val oddThickness = thickness and 1
        thickness = thickness shl (XY_SHIFT - 1)
        if (abs(r) > Math.ulp(1.0)) {
            r = (thickness + oddThickness * XY_ONE * 0.5) / sqrt(r)
            val dpx = cvRound(dy * r)
            val dpy = cvRound(dx * r)
            val pts = listOf(
                Pt(p0.x + dpx, p0.y + dpy),
                Pt(p0.x - dpx, p0.y - dpy),
                Pt(p1.x - dpx, p1.y - dpy),
                Pt(p1.x + dpx, p1.y + dpy),
            )
            fillConvexPoly(img, pts, color, XY_SHIFT)
        }
        var p = p0
        for (i in 0..1) {
            val cx = ((p.x + (XY_ONE shr 1)) shr XY_SHIFT).toInt()
            val cy = ((p.y + (XY_ONE shr 1)) shr XY_SHIFT).toInt()
            fillCircle(img, cx, cy, ((thickness + (XY_ONE shr 1)) shr XY_SHIFT).toInt(), color)
            p = p1
        }
    }

    /** OpenCV's `SinTable`: sin of 0..450 whole degrees as float literals (drawing.cpp), verbatim. */
    private val SIN_TABLE = floatArrayOf(
        0.0000000f, 0.0174524f, 0.0348995f, 0.0523360f, 0.0697565f, 0.0871557f, 0.1045285f, 0.1218693f,
        0.1391731f, 0.1564345f, 0.1736482f, 0.1908090f, 0.2079117f, 0.2249511f, 0.2419219f, 0.2588190f,
        0.2756374f, 0.2923717f, 0.3090170f, 0.3255682f, 0.3420201f, 0.3583679f, 0.3746066f, 0.3907311f,
        0.4067366f, 0.4226183f, 0.4383711f, 0.4539905f, 0.4694716f, 0.4848096f, 0.5000000f, 0.5150381f,
        0.5299193f, 0.5446390f, 0.5591929f, 0.5735764f, 0.5877853f, 0.6018150f, 0.6156615f, 0.6293204f,
        0.6427876f, 0.6560590f, 0.6691306f, 0.6819984f, 0.6946584f, 0.7071068f, 0.7193398f, 0.7313537f,
        0.7431448f, 0.7547096f, 0.7660444f, 0.7771460f, 0.7880108f, 0.7986355f, 0.8090170f, 0.8191520f,
        0.8290376f, 0.8386706f, 0.8480481f, 0.8571673f, 0.8660254f, 0.8746197f, 0.8829476f, 0.8910065f,
        0.8987940f, 0.9063078f, 0.9135455f, 0.9205049f, 0.9271839f, 0.9335804f, 0.9396926f, 0.9455186f,
        0.9510565f, 0.9563048f, 0.9612617f, 0.9659258f, 0.9702957f, 0.9743701f, 0.9781476f, 0.9816272f,
        0.9848078f, 0.9876883f, 0.9902681f, 0.9925462f, 0.9945219f, 0.9961947f, 0.9975641f, 0.9986295f,
        0.9993908f, 0.9998477f, 1.0000000f, 0.9998477f, 0.9993908f, 0.9986295f, 0.9975641f, 0.9961947f,
        0.9945219f, 0.9925462f, 0.9902681f, 0.9876883f, 0.9848078f, 0.9816272f, 0.9781476f, 0.9743701f,
        0.9702957f, 0.9659258f, 0.9612617f, 0.9563048f, 0.9510565f, 0.9455186f, 0.9396926f, 0.9335804f,
        0.9271839f, 0.9205049f, 0.9135455f, 0.9063078f, 0.8987940f, 0.8910065f, 0.8829476f, 0.8746197f,
        0.8660254f, 0.8571673f, 0.8480481f, 0.8386706f, 0.8290376f, 0.8191520f, 0.8090170f, 0.7986355f,
        0.7880108f, 0.7771460f, 0.7660444f, 0.7547096f, 0.7431448f, 0.7313537f, 0.7193398f, 0.7071068f,
        0.6946584f, 0.6819984f, 0.6691306f, 0.6560590f, 0.6427876f, 0.6293204f, 0.6156615f, 0.6018150f,
        0.5877853f, 0.5735764f, 0.5591929f, 0.5446390f, 0.5299193f, 0.5150381f, 0.5000000f, 0.4848096f,
        0.4694716f, 0.4539905f, 0.4383711f, 0.4226183f, 0.4067366f, 0.3907311f, 0.3746066f, 0.3583679f,
        0.3420201f, 0.3255682f, 0.3090170f, 0.2923717f, 0.2756374f, 0.2588190f, 0.2419219f, 0.2249511f,
        0.2079117f, 0.1908090f, 0.1736482f, 0.1564345f, 0.1391731f, 0.1218693f, 0.1045285f, 0.0871557f,
        0.0697565f, 0.0523360f, 0.0348995f, 0.0174524f, 0.0000000f, -0.0174524f, -0.0348995f, -0.0523360f,
        -0.0697565f, -0.0871557f, -0.1045285f, -0.1218693f, -0.1391731f, -0.1564345f, -0.1736482f, -0.1908090f,
        -0.2079117f, -0.2249511f, -0.2419219f, -0.2588190f, -0.2756374f, -0.2923717f, -0.3090170f, -0.3255682f,
        -0.3420201f, -0.3583679f, -0.3746066f, -0.3907311f, -0.4067366f, -0.4226183f, -0.4383711f, -0.4539905f,
        -0.4694716f, -0.4848096f, -0.5000000f, -0.5150381f, -0.5299193f, -0.5446390f, -0.5591929f, -0.5735764f,
        -0.5877853f, -0.6018150f, -0.6156615f, -0.6293204f, -0.6427876f, -0.6560590f, -0.6691306f, -0.6819984f,
        -0.6946584f, -0.7071068f, -0.7193398f, -0.7313537f, -0.7431448f, -0.7547096f, -0.7660444f, -0.7771460f,
        -0.7880108f, -0.7986355f, -0.8090170f, -0.8191520f, -0.8290376f, -0.8386706f, -0.8480481f, -0.8571673f,
        -0.8660254f, -0.8746197f, -0.8829476f, -0.8910065f, -0.8987940f, -0.9063078f, -0.9135455f, -0.9205049f,
        -0.9271839f, -0.9335804f, -0.9396926f, -0.9455186f, -0.9510565f, -0.9563048f, -0.9612617f, -0.9659258f,
        -0.9702957f, -0.9743701f, -0.9781476f, -0.9816272f, -0.9848078f, -0.9876883f, -0.9902681f, -0.9925462f,
        -0.9945219f, -0.9961947f, -0.9975641f, -0.9986295f, -0.9993908f, -0.9998477f, -1.0000000f, -0.9998477f,
        -0.9993908f, -0.9986295f, -0.9975641f, -0.9961947f, -0.9945219f, -0.9925462f, -0.9902681f, -0.9876883f,
        -0.9848078f, -0.9816272f, -0.9781476f, -0.9743701f, -0.9702957f, -0.9659258f, -0.9612617f, -0.9563048f,
        -0.9510565f, -0.9455186f, -0.9396926f, -0.9335804f, -0.9271839f, -0.9205049f, -0.9135455f, -0.9063078f,
        -0.8987940f, -0.8910065f, -0.8829476f, -0.8746197f, -0.8660254f, -0.8571673f, -0.8480481f, -0.8386706f,
        -0.8290376f, -0.8191520f, -0.8090170f, -0.7986355f, -0.7880108f, -0.7771460f, -0.7660444f, -0.7547096f,
        -0.7431448f, -0.7313537f, -0.7193398f, -0.7071068f, -0.6946584f, -0.6819984f, -0.6691306f, -0.6560590f,
        -0.6427876f, -0.6293204f, -0.6156615f, -0.6018150f, -0.5877853f, -0.5735764f, -0.5591929f, -0.5446390f,
        -0.5299193f, -0.5150381f, -0.5000000f, -0.4848096f, -0.4694716f, -0.4539905f, -0.4383711f, -0.4226183f,
        -0.4067366f, -0.3907311f, -0.3746066f, -0.3583679f, -0.3420201f, -0.3255682f, -0.3090170f, -0.2923717f,
        -0.2756374f, -0.2588190f, -0.2419219f, -0.2249511f, -0.2079117f, -0.1908090f, -0.1736482f, -0.1564345f,
        -0.1391731f, -0.1218693f, -0.1045285f, -0.0871557f, -0.0697565f, -0.0523360f, -0.0348995f, -0.0174524f,
        -0.0000000f, 0.0174524f, 0.0348995f, 0.0523360f, 0.0697565f, 0.0871557f, 0.1045285f, 0.1218693f,
        0.1391731f, 0.1564345f, 0.1736482f, 0.1908090f, 0.2079117f, 0.2249511f, 0.2419219f, 0.2588190f,
        0.2756374f, 0.2923717f, 0.3090170f, 0.3255682f, 0.3420201f, 0.3583679f, 0.3746066f, 0.3907311f,
        0.4067366f, 0.4226183f, 0.4383711f, 0.4539905f, 0.4694716f, 0.4848096f, 0.5000000f, 0.5150381f,
        0.5299193f, 0.5446390f, 0.5591929f, 0.5735764f, 0.5877853f, 0.6018150f, 0.6156615f, 0.6293204f,
        0.6427876f, 0.6560590f, 0.6691306f, 0.6819984f, 0.6946584f, 0.7071068f, 0.7193398f, 0.7313537f,
        0.7431448f, 0.7547096f, 0.7660444f, 0.7771460f, 0.7880108f, 0.7986355f, 0.8090170f, 0.8191520f,
        0.8290376f, 0.8386706f, 0.8480481f, 0.8571673f, 0.8660254f, 0.8746197f, 0.8829476f, 0.8910065f,
        0.8987940f, 0.9063078f, 0.9135455f, 0.9205049f, 0.9271839f, 0.9335804f, 0.9396926f, 0.9455186f,
        0.9510565f, 0.9563048f, 0.9612617f, 0.9659258f, 0.9702957f, 0.9743701f, 0.9781476f, 0.9816272f,
        0.9848078f, 0.9876883f, 0.9902681f, 0.9925462f, 0.9945219f, 0.9961947f, 0.9975641f, 0.9986295f,
        0.9993908f, 0.9998477f, 1.0000000f,
    )
}
