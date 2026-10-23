package pl.genor.glassvision

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max

/**
 * "Box glide": results arrive at ~10 Hz, so drawn boxes would jump. Each track (by id) is drawn at a
 * position that chases its newest target with exponential smoothing, `k = 1 - exp(-dt / tau)` per
 * animation frame — frame-rate independent and never overshooting (0 <= k <= 1).
 *
 * New tracks appear at their target, tracks missing from the newest result vanish at once, tracks without
 * an id (raw detections) are never animated. Pure: time comes in as explicit `nowMs`, coordinates are
 * the normalized ones from the wire. Not thread-safe (overlay = main thread).
 */
class BoxAnimator(tauMs: Float = DEFAULT_TAU_MS) {

    /** Smoothing time constant; 0 = off (draw targets as they come). */
    var tauMs: Float = tauMs
        set(value) {
            field = value.coerceAtLeast(0f)
        }

    /** Max remaining distance (normalized) that counts as arrived; the overlay sets ~0.5 px. */
    var epsilon: Float = 0.0005f

    private class Animated(var target: Detection) {
        val box = floatArrayOf(target.box.x1, target.box.y1, target.box.x2, target.box.y2)
        var points = pointsOf(target)
    }

    private var tracks = LinkedHashMap<String, Animated>()
    private var lastStepMs: Long? = null
    private var moving = false

    /** True while some box has not reached its target yet (keep requesting frames). */
    val isMoving: Boolean get() = moving

    fun setTargets(targets: List<Detection>, nowMs: Long) {
        val next = LinkedHashMap<String, Animated>(targets.size * 2)
        targets.forEachIndexed { i, d ->
            val id = d.trackId
            val prev = if (id != null && tauMs > 0f) tracks[id] else null
            val key = id ?: "\u0000$i" // id-less entries never match a previous one
            if (prev == null) {
                next[key] = Animated(d)
            } else {
                prev.target = d
                // Point count changed (different landmark set): nothing sensible to interpolate.
                if (prev.points.size != d.points.size * 2) prev.points = pointsOf(d)
                next[key] = prev
            }
        }
        tracks = next
        // After an idle gap, start timing from now: a stale lastStep would make the first frame jump.
        if (!moving) lastStepMs = nowMs
        moving = tracks.values.any { distance(it) > epsilon }
    }

    /** Advances every box towards its target; returns [isMoving]. */
    fun step(nowMs: Long): Boolean {
        val dt = (nowMs - (lastStepMs ?: nowMs)).coerceAtLeast(0L)
        lastStepMs = nowMs
        val k = if (tauMs <= 0f) 1f else (1.0 - exp(-dt / tauMs.toDouble())).toFloat()
        var anyMoving = false
        for (a in tracks.values) {
            val t = a.target
            a.box[0] += (t.box.x1 - a.box[0]) * k
            a.box[1] += (t.box.y1 - a.box[1]) * k
            a.box[2] += (t.box.x2 - a.box[2]) * k
            a.box[3] += (t.box.y2 - a.box[3]) * k
            t.points.forEachIndexed { i, p ->
                a.points[2 * i] += (p.x - a.points[2 * i]) * k
                a.points[2 * i + 1] += (p.y - a.points[2 * i + 1]) * k
            }
            if (distance(a) <= epsilon) snap(a) else anyMoving = true
        }
        moving = anyMoving
        return anyMoving
    }

    /** What to draw now: the newest targets' data at their current animated geometry. */
    fun current(): List<Detection> = tracks.values.map { a ->
        val t = a.target
        t.copy(
            box = NormBox(a.box[0], a.box[1], a.box[2], a.box[3]),
            points = t.points.mapIndexed { i, p -> p.copy(x = a.points[2 * i], y = a.points[2 * i + 1]) },
        )
    }

    fun clear() {
        tracks = LinkedHashMap()
        moving = false
        lastStepMs = null
    }

    private fun distance(a: Animated): Float {
        val t = a.target
        var d = max(max(abs(t.box.x1 - a.box[0]), abs(t.box.y1 - a.box[1])), max(abs(t.box.x2 - a.box[2]), abs(t.box.y2 - a.box[3])))
        t.points.forEachIndexed { i, p ->
            d = max(d, max(abs(p.x - a.points[2 * i]), abs(p.y - a.points[2 * i + 1])))
        }
        return d
    }

    private fun snap(a: Animated) {
        val t = a.target
        a.box[0] = t.box.x1
        a.box[1] = t.box.y1
        a.box[2] = t.box.x2
        a.box[3] = t.box.y2
        a.points = pointsOf(t)
    }

    companion object {
        const val DEFAULT_TAU_MS = 70f

        private fun pointsOf(d: Detection): FloatArray =
            FloatArray(d.points.size * 2).also { arr ->
                d.points.forEachIndexed { i, p ->
                    arr[2 * i] = p.x
                    arr[2 * i + 1] = p.y
                }
            }
    }
}
