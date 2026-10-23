package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/** On-device mode in the pure screen models, settings and file handling. */
class LocalModeTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private val live = StatsSnapshot(9.8f, 9.6f, 85f, 70.0, linkedMapOf("objects" to 31.0, "pose" to 24.5), 0, 0, 3)
    private val local = ProcessingMode.LOCAL

    @Test
    fun pill_says_on_device_with_fps_and_never_offline() {
        assertEquals(StatusPillModel("On device · 9.6 fps", Tone.GOOD), StatusText.pill(ConnectionState.CONNECTED, false, live, local))
        assertEquals(StatusPillModel("Paused", Tone.IDLE), StatusText.pill(ConnectionState.CONNECTED, true, live, local))
        assertEquals(Tone.BUSY, StatusText.pill(ConnectionState.DISCONNECTED, false, live, local).tone)
        assertEquals(Tone.BUSY, StatusText.pill(ConnectionState.NOT_CONFIGURED, false, live, local).tone)
    }

    @Test
    fun nothing_on_the_phone_opens_the_connection_sheet() {
        for (state in ConnectionState.entries) assertFalse(StatusText.needsSetup(state, local))
        assertTrue(StatusText.needsSetup(ConnectionState.NOT_CONFIGURED))
    }

    @Test
    fun banner_until_the_engine_said_hello() {
        assertEquals(ChipBanner.STARTING_LOCAL, StatusText.banner(ConnectionState.DISCONNECTED, null, local))
        assertEquals(ChipBanner.STARTING_LOCAL, StatusText.banner(ConnectionState.CONNECTED, null, local))
        assertNull(StatusText.banner(ConnectionState.CONNECTED, LocalCatalog.specs, local))
    }

    @Test
    fun hud_shows_engine_latency_and_phone_timings() {
        val rows = StatusText.hudRows(ConnectionState.CONNECTED, LocalBackend.DETAIL, "v0.6.0.3 · CPU", false, live, local)
        val map = rows.associate { it.label to it.value }
        assertEquals("on this phone", map["State"])
        assertEquals("v0.6.0.3 · CPU", map["Engine"])
        assertEquals("85 ms", map["Latency"])
        assertEquals("70.0 ms", map["Engine time"])
        assertEquals("31 ms", map["  objects"])
        assertFalse(rows.any { it.label == "Server" || it.label == "Round trip" })
    }

    @Test
    fun processing_mode_survives_reset_and_round_trips_its_pref() {
        val stream = SnapshotFixtures.MemoryStream(StreamSettings(processing = local, localGpu = true, targetFps = 5))
        stream.resetToDefaults()
        assertEquals(local, stream.settings.processing)
        assertFalse("GPU is a tunable: back to the default", stream.settings.localGpu)
        assertEquals(10, stream.settings.targetFps)
        for (m in ProcessingMode.entries) assertEquals(m, ProcessingMode.fromPref(m.prefValue))
        assertEquals(ProcessingMode.SERVER, ProcessingMode.fromPref(null))
        assertEquals(ProcessingMode.SERVER, ProcessingMode.fromPref("cloud"))
    }

    @Test
    fun local_capture_files_only_from_inside_the_capture_folder() {
        val root = tmp.newFolder("captures")
        val capture = File(root, "20261023-101500-123").apply { mkdirs() }
        val photo = File(capture, "photo.jpg").apply { writeText("jpeg") }
        val secret = tmp.newFile("glass_vision.xml").apply { writeText("token") }

        assertEquals(photo.canonicalFile, CaptureFiles.localSource(photo.toURI().toString(), root))
        assertNull(CaptureFiles.localSource(secret.toURI().toString(), root))
        assertNull(CaptureFiles.localSource("file:${capture.path}/../../glass_vision.xml", root))
        assertNull(CaptureFiles.localSource(File(capture, "missing.png").toURI().toString(), root))
        assertNull(CaptureFiles.localSource("/captures/20261023-101500-123/photo.jpg", root)) // server URL
        assertNull(CaptureFiles.localSource("file:not a uri", root))
        assertNull(CaptureFiles.localSource(capture.toURI().toString(), root)) // a directory
    }

    @Test
    fun local_capture_urls_plan_into_the_same_gallery_folder() {
        val files = CaptureStore.FILES.map { (name, kind) ->
            ServerMessage.CommandResult.ResultFile(name, kind, "file:/data/user/0/pl.genor.glassvision/cache/captures/20261023-101500-123/$name")
        }
        val targets = CaptureFiles.plan(files, "fallback")
        assertEquals(setOf("GlassVision/20261023-101500-123"), targets.map { it.subfolder }.toSet())
        assertEquals(listOf(true, true, true, false, true), targets.map { it.isImage })
    }
}
