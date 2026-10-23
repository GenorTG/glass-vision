package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.min

class TrackStyleTest {
    private fun hueDistance(a: Float, b: Float) = abs(a - b).let { min(it, 360f - it) }

    @Test
    fun pipelines_get_distinct_base_hues_and_labels_stay_near_them() {
        val objects = TrackStyle.hue("objects", 0, "person")
        val faces = TrackStyle.hue("faces", 1, "person")
        val text = TrackStyle.hue("text", 2, "person")
        assertTrue(hueDistance(objects, faces) > 60f)
        assertTrue(hueDistance(faces, text) > 60f)
        assertTrue(hueDistance(objects, text) > 60f)
        // Same pipeline, different labels: within ±24° of each other's family.
        val cup = TrackStyle.hue("objects", 0, "cup")
        assertTrue(hueDistance(objects, cup) <= 48f)
        // Stable and in range, also for pipelines missing from hello.
        assertEquals(objects, TrackStyle.hue("objects", 0, "person"), 0f)
        val unknown = TrackStyle.hue("brand_new", -1, "x")
        assertTrue(unknown in 0f..360f)
    }

    @Test
    fun label_text_has_detail_and_percent() {
        val d = Detection("faces", "Alex", 0.934f, NormBox(0f, 0f, 1f, 1f), detail = "0.71")
        assertEquals("Alex · 0.71 93%", TrackStyle.labelText(d))
        assertEquals("person 50%", TrackStyle.labelText(d.copy(label = "person", detail = null, conf = 0.5f)))
        assertEquals("STOP 80%", TrackStyle.labelText(d.copy(pipeline = "text", label = "STOP", detail = "STOP", conf = 0.8f)))
        assertTrue(TrackStyle.isThin(d.copy(pipeline = "text")))
        assertFalse(TrackStyle.isThin(d))
    }

    @Test
    fun hud_pipeline_times_are_compact_and_slow_ones_linger() {
        assertEquals(
            "objects 8.1 · faces 4.2 · text 120",
            StreamStats.formatPipelineMs(linkedMapOf("objects" to 8.1, "faces" to 4.2, "text" to 120.4)),
        )
        var now = 0L
        val stats = StreamStats { now }
        stats.onDetections(10, 14.0, linkedMapOf("objects" to 8.0, "text" to 120.0), 3)
        now = 100
        stats.onDetections(10, 9.0, linkedMapOf("objects" to 7.0), 2) // OCR didn't run this frame
        assertEquals(linkedMapOf("objects" to 7.0, "text" to 120.0), stats.snapshot().pipelineMs)
        now = 5_050 // text last reported 5.05 s ago → dropped, objects (4.95 s) kept
        assertEquals(mapOf("objects" to 7.0), stats.snapshot().pipelineMs)
    }
}
