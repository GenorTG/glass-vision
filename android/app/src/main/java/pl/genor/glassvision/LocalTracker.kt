package pl.genor.glassvision

import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** One raw result of an on-device pipeline, in pixels of the frame it ran on (the server's `Annotation`). */
data class PixelDetection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val conf: Float,
    val label: String,
    val detail: String? = null,
    /** Keypoints in pixels (x, y, visibility). */
    val points: List<KeyPoint>? = null,
) {
    /** Wire shape (`dets`): normalized, clipped to the frame, like the server's `tracking.normalize`. */
    fun normalized(pipeline: String, width: Int, height: Int): Detection = Detection(
        pipeline = pipeline,
        label = label,
        conf = conf,
        box = NormBox(clip(x1 / width), clip(y1 / height), clip(x2 / width), clip(y2 / height)),
        detail = detail,
        points = points?.map { KeyPoint(clip(it.x / width), clip(it.y / height), it.v) }.orEmpty(),
    )

    companion object {
        fun clip(v: Float): Float = v.coerceIn(0f, 1f)
    }
}

/**
 * Temporal smoothing for one on-device pipeline, with the same feel as the server's `tracking.py`
 * (ByteTrack + hold + label vote) in plain Kotlin:
 * 1. Two-stage greedy IoU matching: detections at or above the user's threshold match first and may start tracks;
 *    lower ones (down to [LOW_CONF]) can only keep an existing track alive. A new track is shown from its second
 *    consecutive update (one-frame ghosts never appear), except on the very first update (like ByteTrack's frame 1).
 * 2. Hold: a missed track stays visible for `holdMs`, moved along its last velocity (`seenMsAgo` > 0 = dimmed).
 *    It can be re-matched (same id) for [TRACK_BUFFER] updates.
 * 3. Per-label score votes that decay each update, and an EMA of the confidence.
 * Ids are stable strings `"<pipeline>:<n>"`. Not thread-safe: the caller serializes [update]/[views].
 */
class LocalTracker(private val pipeline: String) {
    private class Track(
        val id: Int,
        var box: FloatArray,
        var conf: Float,
        var lastSeenMs: Long,
        var detail: String?,
        var points: List<KeyPoint>?,
        var confirmed: Boolean,
    ) {
        val votes = LinkedHashMap<String, Float>()
        /** Box velocity per ms (x1, y1, x2, y2), smoothed. */
        val velocity = FloatArray(4)
        var missedUpdates = 0
        /** Predicted box while lost; null while tracked. */
        var predicted: FloatArray? = null

        fun vote(label: String, score: Float) {
            for (k in votes.keys) votes[k] = votes.getValue(k) * LABEL_DECAY
            votes[label] = (votes[label] ?: 0f) + score
            // Drop labels that decayed to nothing (OCR produces many one-off strings).
            votes.entries.removeAll { it.value <= 1e-3f }
        }

        val label: String get() = votes.maxByOrNull { it.value }?.key.orEmpty()
    }

    private val tracks = ArrayList<Track>()
    private var nextId = 1
    private var updates = 0
    private var width = 1
    private var height = 1

    /**
     * Feeds one frame's detections ([width]×[height] pixels). [newTrackConf]: the user's threshold for pipelines
     * that use the global confidence; null = the pipeline already filtered, accept everything.
     */
    fun update(detections: List<PixelDetection>, width: Int, height: Int, newTrackConf: Float?, nowMs: Long) {
        updates++
        this.width = width
        this.height = height
        val floor = if (newTrackConf != null) LOW_CONF else 0f
        val high = detections.filter { it.conf >= (newTrackConf ?: 0f) }
        val low = detections.filter { it.conf >= floor && it.conf < (newTrackConf ?: 0f) }

        val unmatchedTracks = tracks.toMutableList()
        val matched = HashSet<Track>()
        // Stage 1: high-confidence detections vs every live track (tracked first, then lost: like ByteTrack's pool).
        val remainingHigh = match(unmatchedTracks, high, MATCH_IOU, nowMs, matched)
        // Stage 2: low-confidence detections can only rescue tracked (not lost, not tentative) tracks.
        val rescuable = unmatchedTracks.filter { it.confirmed && it.predicted == null }.toMutableList()
        match(rescuable, low, RESCUE_IOU, nowMs, matched)

        for (t in tracks) {
            if (t in matched) continue
            if (!t.confirmed) {
                t.missedUpdates = TRACK_BUFFER + 1 // a tentative track missed in its second update was a ghost
                continue
            }
            t.missedUpdates++
            val elapsed = (nowMs - t.lastSeenMs).toFloat()
            t.predicted = FloatArray(4) { i -> t.box[i] + t.velocity[i] * elapsed }.also(::clipBox)
        }
        tracks.removeAll { it.missedUpdates > TRACK_BUFFER }

        for (d in remainingHigh) {
            val t = Track(
                id = nextId++, box = d.box(), conf = d.conf, lastSeenMs = nowMs, detail = d.detail,
                points = d.points, confirmed = updates == 1,
            )
            t.vote(d.label, d.conf)
            tracks += t
        }
    }

    /** Tracks to display now; forgets those not seen for longer than [holdMs]. Normalized wire shape. */
    fun views(holdMs: Int, nowMs: Long): List<Detection> {
        tracks.removeAll { it.confirmed && nowMs - it.lastSeenMs > holdMs && nowMs - it.lastSeenMs > 0 }
        return tracks.filter { it.confirmed }.map { t ->
            val held = t.predicted
            val box = held ?: t.box
            Detection(
                pipeline = pipeline,
                label = t.label,
                conf = t.conf,
                box = NormBox(
                    PixelDetection.clip(box[0] / width), PixelDetection.clip(box[1] / height),
                    PixelDetection.clip(box[2] / width), PixelDetection.clip(box[3] / height),
                ),
                trackId = "$pipeline:${t.id}",
                seenMsAgo = if (held != null) (nowMs - t.lastSeenMs).toInt() else 0,
                detail = t.detail,
                points = t.points?.map {
                    KeyPoint(PixelDetection.clip(it.x / width), PixelDetection.clip(it.y / height), it.v)
                }.orEmpty(),
            )
        }
    }

    fun clear() {
        tracks.clear()
    }

    /** Greedy: best IoU pair first. Removes matched tracks from [pool]; returns the unmatched detections. */
    private fun match(
        pool: MutableList<Track>,
        detections: List<PixelDetection>,
        minIou: Float,
        nowMs: Long,
        matched: MutableSet<Track>,
    ): List<PixelDetection> {
        data class Pair3(val iou: Float, val track: Track, val det: Int)
        val pairs = ArrayList<Pair3>()
        for (t in pool) {
            val tb = t.predicted ?: t.box
            detections.forEachIndexed { i, d ->
                val iou = iou(tb, d.box())
                if (iou >= minIou) pairs += Pair3(iou, t, i)
            }
        }
        pairs.sortByDescending { it.iou }
        val usedDets = HashSet<Int>()
        for (p in pairs) {
            if (p.track in matched || p.det in usedDets) continue
            matched += p.track
            usedDets += p.det
            pool.remove(p.track)
            apply(p.track, detections[p.det], nowMs)
        }
        return detections.filterIndexed { i, _ -> i !in usedDets }
    }

    private fun apply(t: Track, d: PixelDetection, nowMs: Long) {
        val box = d.box()
        val dt = (nowMs - t.lastSeenMs).toFloat()
        if (dt > 0) {
            for (i in 0..3) {
                val v = (box[i] - t.box[i]) / dt
                t.velocity[i] = if (t.velocity[i] == 0f) v else t.velocity[i] + VELOCITY_ALPHA * (v - t.velocity[i])
            }
        }
        t.box = box
        t.conf += CONF_ALPHA * (d.conf - t.conf)
        t.lastSeenMs = nowMs
        t.detail = d.detail
        t.points = d.points
        t.predicted = null
        t.missedUpdates = 0
        t.confirmed = true
        t.vote(d.label, d.conf)
    }

    private fun clipBox(b: FloatArray) {
        b[0] = b[0].coerceIn(0f, width.toFloat())
        b[2] = b[2].coerceIn(0f, width.toFloat())
        b[1] = b[1].coerceIn(0f, height.toFloat())
        b[3] = b[3].coerceIn(0f, height.toFloat())
    }

    private fun PixelDetection.box() = floatArrayOf(x1, y1, x2, y2)

    companion object {
        /** Detector floor for global-confidence pipelines (server: `tracking.LOW_CONF`). */
        const val LOW_CONF = 0.1f
        /** How many updates a lost track can still be re-matched with the same id (server: 30 frames). */
        const val TRACK_BUFFER = 30
        /** ByteTrack's match_thresh 0.8 is a cost (1 - IoU): IoU ≥ 0.2 matches in the first stage. */
        const val MATCH_IOU = 0.2f
        /** ByteTrack's second (low-score) stage uses cost 0.5: IoU ≥ 0.5. */
        const val RESCUE_IOU = 0.5f
        const val LABEL_DECAY = 0.8f
        const val CONF_ALPHA = 0.3f
        const val VELOCITY_ALPHA = 0.5f

        fun iou(a: FloatArray, b: FloatArray): Float {
            val ix = max(0f, min(a[2], b[2]) - max(a[0], b[0]))
            val iy = max(0f, min(a[3], b[3]) - max(a[1], b[1]))
            val inter = ix * iy
            val union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return if (union <= 0f) 0f else inter / union
        }

        /** Slow pipelines are held at least 2.5 × max(interval, measured run time), like the server. */
        fun holdFor(holdMs: Int, intervalMs: Int, measuredMs: Double): Int =
            if (intervalMs <= 0) holdMs else max(holdMs, (2.5 * max(intervalMs.toDouble(), measuredMs)).roundToInt())
    }
}
