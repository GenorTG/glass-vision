package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.exp

class BoxAnimatorTest {
    private fun track(id: String?, x: Float, points: List<KeyPoint> = emptyList(), label: String = "person") =
        Detection("objects", label, 0.9f, NormBox(x, 0.2f, x + 0.2f, 0.6f), trackId = id, points = points)

    private fun x1(a: BoxAnimator, id: String) = a.current().first { it.trackId == id }.box.x1

    @Test
    fun new_track_appears_at_its_target() {
        val a = BoxAnimator()
        a.setTargets(listOf(track("objects:1", 0.3f)), nowMs = 1000)
        assertFalse(a.isMoving)
        assertFalse(a.step(1016))
        assertEquals(0.3f, x1(a, "objects:1"), 0f)
    }

    @Test
    fun moved_track_glides_by_exponential_smoothing_without_overshoot_and_converges() {
        val a = BoxAnimator(tauMs = 70f)
        a.setTargets(listOf(track("objects:1", 0.0f)), 0)
        a.setTargets(listOf(track("objects:1", 0.5f)), 100)
        assertTrue(a.isMoving)
        assertEquals(0.0f, x1(a, "objects:1"), 0f) // nothing moves before the first frame

        // First frame 16 ms after the result: k = 1 - exp(-16/70).
        a.step(116)
        val expected = 0.5f * (1f - exp(-16f / 70f))
        assertEquals(expected, x1(a, "objects:1"), 1e-5f)

        var prev = x1(a, "objects:1")
        var t = 116L
        var frames = 0
        while (a.step(t + 16)) {
            t += 16
            frames++
            val x = x1(a, "objects:1")
            assertTrue("monotonic", x >= prev)
            assertTrue("no overshoot", x <= 0.5f)
            prev = x
            assertTrue("converges", frames < 100)
        }
        assertEquals(0.5f, x1(a, "objects:1"), 0f) // snapped exactly once within epsilon
        // ~0.5 normalized → 0.0005 epsilon needs ln(1000)·70 ≈ 480 ms ≈ 30 frames.
        assertTrue("took $frames frames", frames in 20..40)
    }

    @Test
    fun idle_gap_does_not_make_the_first_frame_jump() {
        val a = BoxAnimator(tauMs = 70f)
        a.setTargets(listOf(track("objects:1", 0.0f)), 0)
        a.step(16) // converged (nothing to do) → idle
        a.setTargets(listOf(track("objects:1", 0.4f)), 5000) // next result long after
        a.step(5016)
        assertEquals(0.4f * (1f - exp(-16f / 70f)), x1(a, "objects:1"), 1e-5f)
    }

    @Test
    fun removed_tracks_disappear_at_once_and_new_ones_join() {
        val a = BoxAnimator()
        a.setTargets(listOf(track("objects:1", 0.1f), track("objects:2", 0.5f)), 0)
        a.setTargets(listOf(track("objects:2", 0.6f), track("faces:1", 0.3f, label = "Alex")), 100)
        assertEquals(listOf("objects:2", "faces:1"), a.current().map { it.trackId })
        assertEquals(0.5f, x1(a, "objects:2"), 0f)
        assertEquals(0.3f, x1(a, "faces:1"), 0f)
        assertEquals("Alex", a.current()[1].label) // non-geometry data is always the newest
    }

    @Test
    fun points_glide_with_the_box_and_snap_when_count_changes() {
        val a = BoxAnimator(tauMs = 70f)
        a.setTargets(listOf(track("g:1", 0.0f, listOf(KeyPoint(0f, 0f, 1f), KeyPoint(0.2f, 0.2f, 0.1f)))), 0)
        a.setTargets(listOf(track("g:1", 0.0f, listOf(KeyPoint(0.4f, 0.4f, 1f), KeyPoint(0.2f, 0.2f, 0.9f)))), 100)
        a.step(116)
        val p = a.current().single().points
        assertEquals(0.4f * (1f - exp(-16f / 70f)), p[0].x, 1e-5f)
        assertEquals(0.9f, p[1].v, 0f) // visibility is not animated

        a.setTargets(listOf(track("g:1", 0.0f, listOf(KeyPoint(0.9f, 0.9f, 1f)))), 200)
        assertEquals(KeyPoint(0.9f, 0.9f, 1f), a.current().single().points.single())
    }

    @Test
    fun tau_zero_and_id_less_entries_never_animate() {
        val off = BoxAnimator(tauMs = 0f)
        off.setTargets(listOf(track("objects:1", 0.0f)), 0)
        off.setTargets(listOf(track("objects:1", 0.5f)), 100)
        assertFalse(off.isMoving)
        assertEquals(0.5f, x1(off, "objects:1"), 0f)

        val a = BoxAnimator()
        a.setTargets(listOf(track(null, 0.0f)), 0)
        a.setTargets(listOf(track(null, 0.5f)), 100)
        assertEquals(0.5f, a.current().single().box.x1, 0f)
    }

    @Test
    fun clear_forgets_everything() {
        val a = BoxAnimator()
        a.setTargets(listOf(track("objects:1", 0.1f)), 0)
        a.clear()
        assertTrue(a.current().isEmpty())
        assertFalse(a.step(16))
    }
}
