package pl.genor.glassvision

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * Rule-based pose names from the 17 COCO keypoints of one person: "Standing · Arms up", "Sitting", "T-pose"…
 * A line-by-line port of the server's `glass_vision/pose_names.py` (same thresholds, same order of checks, same
 * labels; read its docstring for the rules). PoseNamesVectorTest checks this port against labels the Python code
 * produced (`src/test/gen/gen_test_vectors.py`), so change both together or not at all.
 *
 * Points are pixel (x, y, visibility) in image coordinates (y down); visibility < [MIN_VIS] = not seen.
 */
object PoseNames {
    const val MIN_VIS = 0.3
    const val SEP = " · "

    private const val NOSE = 0
    private const val L_EYE = 1
    private const val R_EYE = 2
    private const val L_EAR = 3
    private const val R_EAR = 4
    private const val L_SHOULDER = 5
    private const val R_SHOULDER = 6
    private const val L_ELBOW = 7
    private const val R_ELBOW = 8
    private const val L_WRIST = 9
    private const val R_WRIST = 10
    private const val L_HIP = 11
    private const val R_HIP = 12
    private const val L_KNEE = 13
    private const val R_KNEE = 14
    private const val L_ANKLE = 15
    private const val R_ANKLE = 16

    /** (shoulder, elbow, wrist, hip) per side, left first. */
    private val SIDES = listOf(
        intArrayOf(L_SHOULDER, L_ELBOW, L_WRIST, L_HIP),
        intArrayOf(R_SHOULDER, R_ELBOW, R_WRIST, R_HIP),
    )
    private val LEGS = listOf(intArrayOf(L_HIP, L_KNEE, L_ANKLE), intArrayOf(R_HIP, R_KNEE, R_ANKLE))

    private const val SHOULDER_WIDTH_PER_TORSO = 0.75
    private const val HEAD_TO_NECK_PER_TORSO = 0.45

    /** 17 COCO keypoints (pixel x, y, visibility) of one person → a short human-readable pose label. */
    fun name(points: List<KeyPoint>): String = name(points.map { Triple(it.x.toDouble(), it.y.toDouble(), it.v.toDouble()) })

    @JvmName("nameOfTriples")
    fun name(points: List<Triple<Double, Double, Double>>): String {
        val b = Body.of(points)
        if (b.scale == null) return "person"
        val posture = posture(b)
        val arms = when (posture) {
            "Bending over" -> null
            "Lying down" -> arms(b.rotatedUpright(), b)
            else -> arms(b, b)
        }
        return when {
            posture != null && arms != null -> "$posture$SEP$arms"
            posture != null -> posture
            arms != null -> arms
            b.p[L_SHOULDER] != null || b.p[R_SHOULDER] != null -> "Upper body"
            else -> "person"
        }
    }

    private class V(val x: Double, val y: Double) {
        operator fun minus(o: V) = V(x - o.x, y - o.y)
        operator fun plus(o: V) = V(x + o.x, y + o.y)
        operator fun div(d: Double) = V(x / d, y / d)
        fun norm() = hypot(x, y)
        fun dot(o: V) = x * o.x + y * o.y
    }

    private fun mean(vs: List<V>): V = V(vs.sumOf { it.x } / vs.size, vs.sumOf { it.y } / vs.size)

    /** Angle in degrees between v and straight up (0, -1) in image coordinates: 0 = up, 90 = sideways, 180 = down. */
    private fun tilt(v: V): Double {
        val n = v.norm()
        if (n < 1e-9) return 0.0
        return Math.toDegrees(acos((-v.y / n).coerceIn(-1.0, 1.0)))
    }

    /** Inner angle at b of a-b-c in degrees (180 = straight). */
    private fun angle(a: V, b: V, c: V): Double {
        val u = a - b
        val w = c - b
        val nu = u.norm()
        val nw = w.norm()
        if (nu < 1e-9 || nw < 1e-9) return 180.0
        return Math.toDegrees(acos((u.dot(w) / (nu * nw)).coerceIn(-1.0, 1.0)))
    }

    /** Visible keypoints of one person as vectors, plus the torso scale and frame. */
    private class Body(val p: List<V?>, val shoulders: V?, val hips: V?, val head: V?, val scale: Double?) {
        fun rotatedUpright(): Body {
            val s = shoulders!!
            val d = s - hips!!
            val a = atan2(d.x, -d.y) // tilt, signed: + = leaning towards +x
            val c = cos(a)
            val sn = sin(a)
            // [[c, s], [-s, c]] rotates the torso direction onto (0, -1).
            fun rot(q: V): V {
                val r = q - s
                return s + V(c * r.x + sn * r.y, -sn * r.x + c * r.y)
            }
            val np = p.map { it?.let(::rot) }
            return Body(np, mid(np, L_SHOULDER, R_SHOULDER), mid(np, L_HIP, R_HIP), head?.let(::rot), scale)
        }

        companion object {
            fun of(points: List<Triple<Double, Double, Double>>): Body {
                val p = MutableList<V?>(17) { null }
                points.take(17).forEachIndexed { i, (x, y, v) ->
                    if (v >= MIN_VIS && x.isFinite() && y.isFinite()) p[i] = V(x, y)
                }
                val shoulders = mid(p, L_SHOULDER, R_SHOULDER)
                val hips = mid(p, L_HIP, R_HIP)
                val headPts = listOf(NOSE, L_EYE, R_EYE, L_EAR, R_EAR).mapNotNull { p[it] }
                val head = if (headPts.isEmpty()) null else mean(headPts)
                return Body(p, shoulders, hips, head, scaleOf(p, shoulders, hips, head))
            }

            fun mid(p: List<V?>, a: Int, b: Int): V? {
                val pa = p[a]
                val pb = p[b]
                if (pa != null && pb != null) return (pa + pb) / 2.0
                return pa ?: pb
            }

            private fun scaleOf(p: List<V?>, shoulders: V?, hips: V?, head: V?): Double? {
                if (shoulders != null && hips != null) {
                    val t = (shoulders - hips).norm()
                    if (t > 1e-6) return t
                }
                val estimates = mutableListOf<Double>()
                val ls = p[L_SHOULDER]
                val rs = p[R_SHOULDER]
                if (ls != null && rs != null) estimates += (ls - rs).norm() / SHOULDER_WIDTH_PER_TORSO
                if (head != null && shoulders != null) estimates += (head - shoulders).norm() / HEAD_TO_NECK_PER_TORSO
                val best = estimates.maxOrNull() ?: 0.0
                return if (best > 1e-6) best else null
            }
        }
    }

    private fun posture(b: Body): String? {
        val shoulders = b.shoulders ?: return null
        val hips = b.hips ?: return null
        val t = b.scale ?: return null
        val theta = tilt(shoulders - hips)

        val knees = LEGS.mapNotNull { b.p[it[1]] }
        val ankles = LEGS.mapNotNull { b.p[it[2]] }
        val foot = if (ankles.isNotEmpty()) mean(ankles) else if (knees.isNotEmpty()) mean(knees) else null
        val legsDown = foot != null && tilt(hips - foot) < 30 && foot.y - hips.y > 0.5 * t

        if (theta > 45 && legsDown) return "Bending over"
        if (theta in 60.0..120.0) return "Lying down"
        if (theta > 60 || knees.isEmpty()) return null

        var kneeling = 0
        var standing = 0
        var squatting = 0
        var sitting = 0
        for ((hipI, kneeI, ankleI) in LEGS.map { Triple(it[0], it[1], it[2]) }) {
            val hip = b.p[hipI] ?: hips
            val knee = b.p[kneeI] ?: continue
            val ankle = b.p[ankleI]
            val thighDrop = (knee.y - hip.y) / t
            if (ankle == null) {
                if (thighDrop > 0.6) standing++ else if (thighDrop < 0.45) sitting++
                continue
            }
            val shinDrop = (ankle.y - knee.y) / t
            val kneeAngle = angle(hip, knee, ankle)
            when {
                shinDrop < 0.3 && thighDrop > 0.45 -> kneeling++
                thighDrop > 0.6 && kneeAngle > 140 -> standing++
                (ankle.y - hip.y) / t < 0.75 && shinDrop >= 0.3 -> squatting++
                thighDrop < 0.45 -> sitting++
                else -> standing++ // half-bent, thigh mostly down: walking/leaning rather than sitting
            }
        }
        if (kneeling > 0) return "Kneeling" // one knee down is enough
        // Python's max() over an insertion-ordered dict: the first maximum wins.
        val votes = listOf("Squatting" to squatting, "Sitting" to sitting, "Standing" to standing)
        var best = votes[0]
        for (v in votes) if (v.second > best.second) best = v
        return if (best.second > 0) best.first else null
    }

    private fun armStraight(b: Body, s: Int, e: Int, w: Int): Boolean {
        val ps = b.p[s] ?: return false
        val pe = b.p[e] ?: return false
        val pw = b.p[w] ?: return false
        return angle(ps, pe, pw) > 150
    }

    /** [b] = the body in the frame used for up/down; [image] = the same body in image coordinates (directions). */
    private fun arms(b: Body, image: Body): String? {
        val t = b.scale ?: return null
        if (b.p[L_SHOULDER] == null && b.p[R_SHOULDER] == null) return null
        val wrists = SIDES.filter { b.p[it[0]] != null && b.p[it[2]] != null }

        b.head?.let { head ->
            val onHead = wrists.count { side ->
                val w = b.p[side[2]]!!
                (w - head).norm() < 0.45 * t && w.y < head.y
            }
            if (onHead > 0) return if (onHead == 2) "Hands on head" else "Hand on head"
        }

        val raised = wrists.filter { side -> b.p[side[0]]!!.y - b.p[side[2]]!!.y > 0.3 * t }
        if (raised.size == 2) return "Arms up"

        val midX = b.shoulders?.x
        val horizontal = mutableListOf<Pair<Int, Int>>() // (shoulder, wrist) of straight, horizontal, long arms
        for (side in wrists) {
            val (s, e, w) = Triple(side[0], side[1], side[2])
            val d = b.p[w]!! - b.p[s]!!
            if (armStraight(b, s, e, w) && d.norm() > 0.6 * t && abs(90 - tilt(d)) < 35) horizontal += s to w
        }
        if (horizontal.size == 2 && midX != null) {
            val outward = horizontal.all { (s, w) ->
                (b.p[w]!!.x - midX) * (b.p[s]!!.x - midX) > 0 && abs(90 - tilt(b.p[w]!! - b.p[s]!!)) < 25
            }
            if (outward) return "T-pose"
        }

        val hips = b.hips
        if (wrists.size == 2 && midX != null && hips != null) {
            val mx: Double = midX
            val top = b.shoulders!!.y
            val bottom = hips.y
            val crossed = wrists.all { side ->
                val w = b.p[side[2]]!!
                (w.x - mx) * (b.p[side[0]]!!.x - mx) < 0 && w.y >= top - 0.1 * t && w.y <= bottom + 0.1 * t
            }
            if (crossed) return "Arms crossed"

            fun onHip(side: IntArray): Boolean {
                val hip = b.p[side[3]] ?: hips
                val elbow = b.p[side[1]] ?: return false
                val w = b.p[side[2]]!!
                if ((w - hip).norm() > 0.4 * t) return false
                val outElbow = abs(elbow.x - mx) > abs(w.x - mx)
                return outElbow && angle(b.p[side[0]]!!, elbow, w) < 130
            }
            if (wrists.all(::onHip)) return "Hands on hips"
        }

        val directions = horizontal.map { (s, w) -> if (image.p[w]!!.x < image.p[s]!!.x) "Pointing left" else "Pointing right" }.toSet()
        if (directions.size == 1) return directions.first() // one straight horizontal arm, or both towards the same side

        if (raised.size == 1) return "One arm up"
        return null
    }
}
