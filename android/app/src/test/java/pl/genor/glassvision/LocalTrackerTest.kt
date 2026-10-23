package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** The server's `tests/test_tracking.py` scenarios, on the on-device [LocalTracker]: same expectations. */
class LocalTrackerTest {
    private val w = 640
    private val h = 480
    private val frameMs = 100L // 10 fps

    /** dets: (x offset px, label, conf) of a 100x200 box at y=100. */
    private fun frame(vararg dets: Triple<Float, String, Float>) =
        dets.map { (x, label, conf) -> PixelDetection(x, 100f, x + 100f, 300f, conf, label) }

    private fun run(frames: List<List<PixelDetection>>, conf: Float? = 0.35f, holdMs: Int = 300): List<List<Detection>> {
        val tracker = LocalTracker("objects")
        return frames.mapIndexed { i, dets ->
            tracker.update(dets, w, h, conf, i * frameMs)
            tracker.views(holdMs, i * frameMs)
        }
    }

    @Test
    fun two_missed_frames_in_ten_do_not_flicker() {
        val frames = (0 until 10).map { i -> if (i == 4 || i == 7) frame() else frame(Triple(100f + i * 3, "person", 0.8f)) }
        val out = run(frames)
        assertTrue(out.all { it.size == 1 })
        assertEquals(setOf(out[0][0].trackId), out.map { it[0].trackId }.toSet()) // one stable identity
        assertTrue(out[0][0].trackId!!.startsWith("objects:"))
        assertEquals(frameMs.toInt(), out[4][0].seenMsAgo)
        assertEquals(0, out[5][0].seenMsAgo)
    }

    @Test
    fun confidence_dip_below_threshold_keeps_track() {
        val confs = listOf(0.8f, 0.8f, 0.8f, 0.25f, 0.2f, 0.8f, 0.8f)
        val out = run(confs.map { frame(Triple(100f, "person", it)) }, holdMs = 0)
        assertTrue(out.all { it.size == 1 && it[0].seenMsAgo == 0 })
        assertTrue(out[4][0].conf > 0.4f) // EMA, not the raw 0.2
    }

    @Test
    fun object_gone_disappears_after_hold() {
        val out = run(List(5) { frame(Triple(100f, "person", 0.9f)) } + List(6) { frame() }, holdMs = 300)
        assertEquals(listOf(1, 1, 1, 0, 0, 0), out.drop(5).map { it.size })
    }

    @Test
    fun one_frame_ghost_is_not_shown() {
        val base = frame(Triple(100f, "person", 0.9f))
        val frames = List(3) { base } + listOf(base + frame(Triple(400f, "cup", 0.6f))) + List(3) { base }
        val out = run(frames)
        assertTrue(out.all { it.size == 1 && it[0].label == "person" })
    }

    @Test
    fun label_vote_resists_single_frame_flip() {
        val labels = listOf("cup", "cup", "cup", "cup", "bowl", "cup", "cup")
        val out = run(labels.map { frame(Triple(100f, it, 0.7f)) })
        assertEquals(List(7) { "cup" }, out.map { it[0].label })
    }

    @Test
    fun prefiltered_pipeline_accepts_low_scores_and_carries_payload() {
        // conf=null: the pipeline already filtered (faces, OCR, gestures); low scores must still track.
        val dets = listOf(
            PixelDetection(100f, 100f, 200f, 300f, 0.15f, "Alex", detail = "0.71", points = listOf(KeyPoint(150f, 150f, 1f))),
        )
        val out = run(listOf(dets, dets, dets), conf = null)
        assertTrue(out.all { it.size == 1 })
        val view = out.last()[0]
        assertEquals("Alex" to "0.71", view.label to view.detail)
        assertEquals(listOf(KeyPoint(150f / w, 150f / h, 1f)), view.points)
    }

    @Test
    fun second_object_appears_from_its_second_frame_with_its_own_id() {
        val a = Triple(100f, "person", 0.9f)
        val b = Triple(400f, "dog", 0.9f)
        val out = run(listOf(frame(a), frame(a, b), frame(a, b), frame(a, b)))
        assertEquals(listOf(1, 1, 2, 2), out.map { it.size })
        assertEquals(2, out[3].map { it.trackId }.toSet().size)
    }

    @Test
    fun below_floor_detections_are_ignored() {
        val out = run(List(3) { frame(Triple(100f, "person", 0.05f)) })
        assertTrue(out.all { it.isEmpty() })
    }

    @Test
    fun slow_pipelines_hold_at_least_two_and_a_half_runs() {
        assertEquals(300, LocalTracker.holdFor(300, 0, 50.0))
        assertEquals(1250, LocalTracker.holdFor(300, 500, 120.0))
        assertEquals(2000, LocalTracker.holdFor(300, 500, 800.0)) // a slow phone: 2.5 × measured
    }
}
