package pl.genor.glassvision

import java.math.BigDecimal
import java.math.MathContext
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.sqrt

/** A keypoint normalized to the canvas (0..1). */
data class NormPoint(val x: Double, val y: Double)

/**
 * One person for the OpenPose export: OpenPose-18 [body] (null = not detected) and the person's own hands
 * (MediaPipe-21 order = OpenPose hand order), null when not found.
 */
data class OpenPosePerson(
    val body: List<NormPoint?>,
    val leftHand: List<NormPoint>? = null,
    val rightHand: List<NormPoint>? = null,
)

/**
 * OpenPose renders + JSON for ComfyUI ControlNet: a port of the server's `glass_vision/openpose.py` (itself
 * comfyui_controlnet_aux's renderer, commit a587a4c) and of `assign_hands` from `pipelines/pose.py`. Same constants,
 * same drawing calls (through [CvDraw], OpenCV's own algorithms), same JSON bytes as `json.dumps`, so a capture made
 * on the phone is interchangeable with one made by the server. OpenPoseVectorTest checks it against renders and
 * JSON the Python code produced. Canvases are RGB, like the reference.
 */
object OpenPose {
    /** DWPose: keypoints below this are "not detected". */
    const val MIN_CONF = 0.3
    /** util.py `eps`: hand points at the canvas origin count as missing. */
    private const val EPS = 0.01

    /** OpenPose-18 index → COCO-17 index (-1 = neck, computed). DWPose's mmpose_idx/openpose_idx remap. */
    val OPENPOSE_FROM_COCO = intArrayOf(0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3)

    /** util.py draw_bodypose, verbatim: 1-based OpenPose-18 index pairs. */
    val LIMB_SEQ = listOf(
        2 to 3, 2 to 6, 3 to 4, 4 to 5,
        6 to 7, 7 to 8, 2 to 9, 9 to 10,
        10 to 11, 2 to 12, 12 to 13, 13 to 14,
        2 to 1, 1 to 15, 15 to 17, 1 to 16,
        16 to 18,
    )

    /** One RGB colour per limb and per joint, verbatim. */
    val COLORS = listOf(
        intArrayOf(255, 0, 0), intArrayOf(255, 85, 0), intArrayOf(255, 170, 0), intArrayOf(255, 255, 0),
        intArrayOf(170, 255, 0), intArrayOf(85, 255, 0), intArrayOf(0, 255, 0), intArrayOf(0, 255, 85),
        intArrayOf(0, 255, 170), intArrayOf(0, 255, 255), intArrayOf(0, 170, 255), intArrayOf(0, 85, 255),
        intArrayOf(0, 0, 255), intArrayOf(85, 0, 255), intArrayOf(170, 0, 255), intArrayOf(255, 0, 255),
        intArrayOf(255, 0, 170), intArrayOf(255, 0, 85),
    )
    private const val STICKWIDTH = 4
    private const val LIMB_ALPHA = 0.6
    private const val JOINT_RADIUS = 4

    /** util.py draw_handpose, verbatim. */
    val HAND_EDGES = listOf(
        0 to 1, 1 to 2, 2 to 3, 3 to 4, 0 to 5, 5 to 6, 6 to 7, 7 to 8, 0 to 9, 9 to 10,
        10 to 11, 11 to 12, 0 to 13, 13 to 14, 14 to 15, 15 to 16, 0 to 17, 17 to 18, 18 to 19, 19 to 20,
    )
    private val HAND_POINT_COLOR = RgbRaster.rgb(0, 0, 255)
    private const val HAND_LINE_THICKNESS = 2
    private const val HAND_POINT_RADIUS = 4

    /** COCO wrist/elbow indices for hand assignment. */
    private const val L_ELBOW = 7
    private const val R_ELBOW = 8
    private const val L_WRIST = 9
    private const val R_WRIST = 10

    /** `matplotlib.colors.hsv_to_rgb([ie / len(edges), 1, 1]) * 255` (= colorsys), as cv2 stores it (rounded). */
    fun handEdgeColor(index: Int): Int {
        val (r, g, b) = hsvToRgb(index / HAND_EDGES.size.toDouble(), 1.0, 1.0)
        return RgbRaster.rgb(CvDraw.channel(r * 255), CvDraw.channel(g * 255), CvDraw.channel(b * 255))
    }

    /** Python's colorsys.hsv_to_rgb. */
    private fun hsvToRgb(h: Double, s: Double, v: Double): Triple<Double, Double, Double> {
        if (s == 0.0) return Triple(v, v, v)
        val i = (h * 6.0).toInt() // int() truncates; h >= 0 here
        val f = h * 6.0 - i
        val p = v * (1.0 - s)
        val q = v * (1.0 - s * f)
        val t = v * (1.0 - s * (1.0 - f))
        return when (i % 6) {
            0 -> Triple(v, t, p)
            1 -> Triple(q, v, p)
            2 -> Triple(p, v, t)
            3 -> Triple(p, q, v)
            4 -> Triple(t, p, v)
            else -> Triple(v, p, q)
        }
    }

    /** 17 COCO keypoints (pixel x, y, confidence) → OpenPose-18, normalized; < [MIN_CONF] → null. */
    fun cocoToOpenPose(points: List<Triple<Double, Double, Double>>, width: Int, height: Int): List<NormPoint?> =
        OPENPOSE_FROM_COCO.map { coco ->
            if (coco >= 0) {
                val (x, y, c) = points[coco]
                if (c >= MIN_CONF) NormPoint(x / width, y / height) else null
            } else {
                val ls = points[5]
                val rs = points[6]
                // DWPose: neck = mean of the shoulders, shown only if both are > 0.3.
                if (ls.third > MIN_CONF && rs.third > MIN_CONF) {
                    NormPoint((ls.first + rs.first) / 2 / width, (ls.second + rs.second) / 2 / height)
                } else {
                    null
                }
            }
        }

    /** A person's left or right hand in [HandAssignment.assigned]. */
    enum class Side { LEFT, RIGHT }

    data class HandAssignment(val assigned: Map<Pair<Int, Side>, Int>, val loose: List<Int>)

    /**
     * pose.py `assign_hands`: greedy nearest-wrist matching. [people] = COCO keypoints in pixels, [handWrists] =
     * each hand's wrist landmark (pixels), [boxes] = (x1, y1, x2, y2) per person. A hand only attaches to a wrist
     * within max(forearm length, 10% of that person's box height) of the hand's wrist.
     */
    fun assignHands(
        people: List<List<Triple<Double, Double, Double>>>,
        handWrists: List<Pair<Double, Double>>,
        boxes: List<DoubleArray>,
    ): HandAssignment {
        data class Cand(val d: Double, val hand: Int, val person: Int, val side: Side)
        val pairs = mutableListOf<Cand>()
        people.forEachIndexed { pi, kp ->
            val boxH = boxes[pi][3] - boxes[pi][1]
            for ((side, wristI, elbowI) in listOf(Triple(Side.LEFT, L_WRIST, L_ELBOW), Triple(Side.RIGHT, R_WRIST, R_ELBOW))) {
                val (wx, wy, wv) = kp[wristI]
                if (wv < MIN_CONF) continue
                val (ex, ey, ev) = kp[elbowI]
                val limit = maxOf(if (ev >= MIN_CONF) hypot(wx - ex, wy - ey) else 0.0, 0.1 * boxH)
                handWrists.forEachIndexed { hi, (hx, hy) ->
                    val d = hypot(hx - wx, hy - wy)
                    if (d <= limit) pairs += Cand(d, hi, pi, side)
                }
            }
        }
        // Python sorts the tuples (d, hand, person, side): ties fall through to the next fields ("left" < "right").
        pairs.sortWith(compareBy<Cand>({ it.d }, { it.hand }, { it.person }, { it.side.ordinal }))
        val assigned = LinkedHashMap<Pair<Int, Side>, Int>()
        val used = HashSet<Int>()
        for (c in pairs) {
            if (c.hand in used || (c.person to c.side) in assigned) continue
            assigned[c.person to c.side] = c.hand
            used += c.hand
        }
        return HandAssignment(assigned, handWrists.indices.filter { it !in used })
    }

    /** util.py draw_bodypose (stick_scale 1) on an RGB canvas, in place. */
    fun drawBodypose(canvas: RgbRaster, keypoints: List<NormPoint?>) {
        val h = canvas.height.toDouble()
        val w = canvas.width.toDouble()
        LIMB_SEQ.forEachIndexed { i, (k1, k2) ->
            if (i >= COLORS.size) return@forEachIndexed
            val p1 = keypoints[k1 - 1] ?: return@forEachIndexed
            val p2 = keypoints[k2 - 1] ?: return@forEachIndexed
            val y0 = p1.x * w
            val y1 = p2.x * w
            val x0 = p1.y * h
            val x1 = p2.y * h
            val mX = (x0 + x1) / 2
            val mY = (y0 + y1) / 2
            val length = sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1))
            val angle = Math.toDegrees(atan2(x0 - x1, y0 - y1))
            val polygon = CvDraw.ellipse2Poly(mY.toInt(), mX.toInt(), (length / 2).toInt(), STICKWIDTH, angle.toInt(), 0, 360, 1)
            val c = COLORS[i]
            // [int(float(c) * 0.6) for c in color]
            val color = RgbRaster.rgb((c[0] * LIMB_ALPHA).toInt(), (c[1] * LIMB_ALPHA).toInt(), (c[2] * LIMB_ALPHA).toInt())
            CvDraw.fillConvexPoly(canvas, polygon, color)
        }
        keypoints.forEachIndexed { i, p ->
            if (p == null || i >= COLORS.size) return@forEachIndexed
            val c = COLORS[i]
            CvDraw.fillCircle(canvas, (p.x * w).toInt(), (p.y * h).toInt(), JOINT_RADIUS, RgbRaster.rgb(c[0], c[1], c[2]))
        }
    }

    /** util.py draw_handpose on an RGB canvas, in place. */
    fun drawHandpose(canvas: RgbRaster, keypoints: List<NormPoint>?) {
        if (keypoints.isNullOrEmpty()) return
        val h = canvas.height
        val w = canvas.width
        HAND_EDGES.forEachIndexed { ie, (e1, e2) ->
            val x1 = (keypoints[e1].x * w).toInt()
            val y1 = (keypoints[e1].y * h).toInt()
            val x2 = (keypoints[e2].x * w).toInt()
            val y2 = (keypoints[e2].y * h).toInt()
            if (x1 > EPS && y1 > EPS && x2 > EPS && y2 > EPS) {
                CvDraw.line(canvas, x1, y1, x2, y2, handEdgeColor(ie), HAND_LINE_THICKNESS)
            }
        }
        for (k in keypoints) {
            val x = (k.x * w).toInt()
            val y = (k.y * h).toInt()
            if (x > EPS && y > EPS) CvDraw.fillCircle(canvas, x, y, HAND_POINT_RADIUS, HAND_POINT_COLOR)
        }
    }

    /**
     * __init__.py draw_poses: a black canvas (or a copy of [background]) with every body, then (if [drawHands]) its
     * hands; [extraHands] are hands attached to no body.
     */
    fun drawPoses(
        people: List<OpenPosePerson>,
        height: Int,
        width: Int,
        drawHands: Boolean = true,
        extraHands: List<List<NormPoint>> = emptyList(),
        background: RgbRaster? = null,
    ): RgbRaster {
        val canvas = background?.copy() ?: RgbRaster(width, height)
        for (person in people) {
            drawBodypose(canvas, person.body)
            if (drawHands) {
                drawHandpose(canvas, person.leftHand)
                drawHandpose(canvas, person.rightHand)
            }
        }
        if (drawHands) extraHands.forEach { drawHandpose(canvas, it) }
        return canvas
    }

    /**
     * __init__.py encode_poses_as_dict, serialized exactly like the server's `json.dumps` (key order, ", "/": "
     * separators, Python float repr): normalized x/y, confidence 1.0 for detected points, 0, 0, 0 for missing
     * ones, no face model (`[]`), a hand that wasn't found is `[]`.
     */
    fun encodePoseJson(people: List<OpenPosePerson>, height: Int, width: Int): String {
        fun body(keypoints: List<NormPoint?>): List<Double> =
            keypoints.flatMap { k -> if (k != null) listOf(k.x, k.y, 1.0) else listOf(0.0, 0.0, 0.0) }

        fun hand(keypoints: List<NormPoint>?): List<Double> =
            keypoints.orEmpty().flatMap { listOf(it.x, it.y, 1.0) }

        fun list(values: List<Double>) = values.joinToString(", ", "[", "]", transform = PyJson::float)

        val persons = people.joinToString(", ", "[", "]") { p ->
            "{\"pose_keypoints_2d\": ${list(body(p.body))}, \"face_keypoints_2d\": [], " +
                "\"hand_left_keypoints_2d\": ${list(hand(p.leftHand))}, \"hand_right_keypoints_2d\": ${list(hand(p.rightHand))}}"
        }
        return "{\"people\": $persons, \"canvas_height\": $height, \"canvas_width\": $width}"
    }
}

/** Number formatting exactly as Python's `json.dumps` writes floats (`float.__repr__`). */
object PyJson {
    /** Shortest digits that round-trip (Python repr), fixed notation for 1e-4 <= |v| < 1e16, else scientific. */
    fun float(v: Double): String {
        if (v.isNaN()) return "NaN"
        if (v.isInfinite()) return if (v > 0) "Infinity" else "-Infinity"
        if (v == 0.0) return if (1.0 / v < 0) "-0.0" else "0.0"
        var digits: BigDecimal? = null
        for (precision in 1..17) {
            val candidate = BigDecimal(v).round(MathContext(precision))
            if (candidate.toDouble() == v) {
                digits = candidate
                break
            }
        }
        val d = (digits ?: BigDecimal(v)).stripTrailingZeros()
        val unscaled = d.unscaledValue().abs().toString()
        // Decimal exponent of the first digit: v = 0.d1d2… × 10^(exp + 1).
        val exp = unscaled.length - 1 - d.scale()
        val sign = if (v < 0) "-" else ""
        return if (exp < -4 || exp >= 16) {
            val mantissa = if (unscaled.length == 1) unscaled else unscaled[0] + "." + unscaled.substring(1)
            val e = if (exp < 0) "-" + (-exp).toString().padStart(2, '0') else "+" + exp.toString().padStart(2, '0')
            "$sign${mantissa}e$e"
        } else if (exp < 0) {
            sign + "0." + "0".repeat(-exp - 1) + unscaled
        } else if (unscaled.length <= exp + 1) {
            sign + unscaled + "0".repeat(exp + 1 - unscaled.length) + ".0"
        } else {
            sign + unscaled.substring(0, exp + 1) + "." + unscaled.substring(exp + 1)
        }
    }
}
