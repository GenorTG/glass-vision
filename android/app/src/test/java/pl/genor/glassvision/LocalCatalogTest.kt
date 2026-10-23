package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The on-device `hello` against the live server's (captured hello_v3.json) and the BlazePose → COCO mapping. */
class LocalCatalogTest {
    private val server = SnapshotFixtures.hello.pipelines.associateBy { it.id }

    @Test
    fun ids_and_skeletons_match_the_server() {
        val ids = LocalCatalog.specs.map { it.id }
        assertEquals(listOf("objects", "gestures", "pose", "faces", "text"), ids)
        assertFalse("open_vocab needs a 570 MB text encoder", "open_vocab" in ids)
        for (spec in LocalCatalog.specs) assertTrue(spec.id in server)
        assertEquals(server.getValue("gestures").skeleton, LocalCatalog.HAND_SKELETON)
        assertEquals(server.getValue("pose").skeleton, LocalCatalog.COCO_SKELETON)
    }

    @Test
    fun shared_options_and_actions_keep_the_server_keys() {
        fun keys(id: String, source: Map<String, PipelineSpec>) = source.getValue(id).options.map { it.key }.toSet()
        val local = LocalCatalog.specs.associateBy { it.id }
        // Same keys where the feature is the same, so a choice made in one mode carries over to the other.
        assertEquals(keys("gestures", server), keys("gestures", local))
        assertTrue("names" in keys("pose", local))
        assertEquals(server.getValue("pose").actions.map { it.id }, local.getValue("pose").actions.map { it.id })
        assertTrue(local.getValue("faces").actions.isEmpty()) // no enroll/list on the phone
        assertTrue("server" in local.getValue("faces").description)
        assertEquals(LocalCatalog.TEXT_INTERVAL_MS, local.getValue("text").intervalMs)
        assertTrue(local.getValue("objects").defaultEnabled)
    }

    @Test
    fun models_follow_the_choice_options() {
        val opt = LocalCatalog.OPT_MODEL
        assertEquals(LocalModel.EFFICIENTDET_LITE0, LocalCatalog.modelFor("objects", emptyMap()))
        assertEquals(LocalModel.EFFICIENTDET_LITE2, LocalCatalog.modelFor("objects", mapOf(opt to OptionValue.Str(LocalCatalog.OBJECTS_ACCURATE))))
        assertEquals(LocalModel.POSE_LITE, LocalCatalog.modelFor("pose", mapOf(opt to OptionValue.Str(LocalCatalog.POSE_LITE))))
        assertEquals(LocalModel.POSE_FULL, LocalCatalog.modelFor("pose", mapOf(opt to OptionValue.Str(LocalCatalog.POSE_FULL))))
        assertNull(LocalCatalog.modelFor("text", emptyMap())) // ML Kit's model is bundled
    }

    @Test
    fun model_urls_are_pinned_google_originals() {
        for (m in LocalModel.entries) {
            assertTrue(m.url, m.url.startsWith("https://storage.googleapis.com/mediapipe-models/"))
            assertTrue(m.url, "/1/" in m.url && m.url.endsWith(m.fileName))
            assertEquals(64, m.sha256.length)
            assertTrue(m.bytes > 100_000)
        }
    }

    @Test
    fun hello_carries_statuses_and_device() {
        val hello = LocalCatalog.hello("0.6.0.3", LocalBackend.device(gpu = true), mapOf("pose" to (PipelineStatus.ERROR to "No internet")))
        assertEquals(Protocol.VERSION, hello.protocol)
        assertEquals("v0.6.0.3 · GPU", StatusText.serverInfo(hello.version, hello.device))
        val pose = hello.pipelines.first { it.id == "pose" }
        assertEquals(PipelineStatus.ERROR to "No internet", pose.status to pose.statusMessage)
        assertEquals(PipelineStatus.NOT_LOADED, hello.pipelines.first { it.id == "objects" }.status)
    }

    @Test
    fun blazepose_maps_to_coco_by_name() {
        // Landmark i at x = i / 100: the COCO point tells which BlazePose landmark it came from.
        val lms = List(33) { KeyPoint(it / 100f, 0.5f, 0.9f) }
        val coco = BlazePose.toCoco(lms, 100, 10)
        // nose, l/r eye, l/r ear, l/r shoulder, l/r elbow, l/r wrist, l/r hip, l/r knee, l/r ankle
        val expected = listOf(0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
        assertEquals(expected, coco.map { Math.round(it.x) })
        assertTrue(coco.all { it.y == 5f && it.v == 0.9f })
    }

    @Test
    fun blazepose_box_pads_visible_points_and_clips() {
        val lms = List(33) { i -> if (i < 2) KeyPoint(0.2f + 0.6f * i, 0.1f + 0.8f * i, 1f) else KeyPoint(0.5f, 0.5f, 0.1f) }
        val box = BlazePose.box(lms, 100, 200)!!
        assertEquals(listOf(14f, 4f, 86f, 196f), box.map { Math.round(it * 10) / 10f })
        assertNull(BlazePose.box(List(33) { KeyPoint(0.5f, 0.5f, 0f) }, 100, 100))
    }
}
