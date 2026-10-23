package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ScreenModelsTest {
    private val hello: ServerMessage.Hello by lazy {
        val text = requireNotNull(javaClass.classLoader?.getResource("hello_v3.json")).readText()
        Protocol.parse(text) as ServerMessage.Hello
    }

    private val stats = StatsSnapshot(10f, 9.84f, 47.6f, 14.25, linkedMapOf("objects" to 8.14, "text" to 55.4), 1, 2, 4)

    @Test
    fun chip_labels_keep_short_names_and_drop_details_of_long_ones() {
        assertEquals("Objects", PipelineLabels.short("Objects (80 everyday classes)"))
        assertEquals("Text (OCR)", PipelineLabels.short("Text (OCR)"))
        assertEquals("Hand gestures", PipelineLabels.short("Hand gestures"))
        assertEquals("Find by text", PipelineLabels.short("  Find by text "))
        val long = PipelineLabels.short("Supercalifragilistic recognizer")
        assertEquals(PipelineLabels.MAX_CHIP_CHARS, long.length)
        assertTrue(long.endsWith("…"))
        // The real server's names all fit a chip.
        hello.pipelines.forEach { assertTrue(it.name, PipelineLabels.short(it.name).length <= PipelineLabels.MAX_CHIP_CHARS) }
    }

    @Test
    fun chip_visual_per_status_and_switch() {
        assertEquals(ChipVisual.OFF, ModeChips.visualOf(PipelineStatus.READY, enabled = false))
        assertEquals(ChipVisual.OFF, ModeChips.visualOf(PipelineStatus.NOT_LOADED, enabled = false))
        assertEquals(ChipVisual.READY, ModeChips.visualOf(PipelineStatus.READY, enabled = true))
        // Old servers send no status: treat as ready.
        assertEquals(ChipVisual.READY, ModeChips.visualOf(PipelineStatus.UNKNOWN, enabled = true))
        assertEquals(ChipVisual.LOADING, ModeChips.visualOf(PipelineStatus.LOADING, enabled = true))
        // Just enabled, the server hasn't started loading yet.
        assertEquals(ChipVisual.LOADING, ModeChips.visualOf(PipelineStatus.NOT_LOADED, enabled = true))
        // Errors show even when off, so the user sees why it doesn't work.
        assertEquals(ChipVisual.ERROR, ModeChips.visualOf(PipelineStatus.ERROR, enabled = false))
        assertEquals(ChipVisual.ERROR, ModeChips.visualOf(PipelineStatus.ERROR, enabled = true))
    }

    @Test
    fun chips_follow_hello_order_defaults_and_user_choices() {
        val settings = PipelineSettings().withEnabled("faces", true).withEnabled("objects", false)
        val specs = hello.pipelines.map { if (it.id == "pose") it.copy(status = PipelineStatus.ERROR, statusMessage = "OOM") else it }
        val chips = ModeChips.of(specs, settings)
        assertEquals(hello.pipelines.map { it.id }, chips.map { it.id })
        val byId = chips.associateBy { it.id }
        assertFalse(byId.getValue("objects").enabled)
        assertTrue(byId.getValue("faces").enabled)
        assertEquals("Objects", byId.getValue("objects").label)
        // Objects has neither options nor actions: no tune affordance.
        assertFalse(byId.getValue("objects").tunable)
        assertTrue(byId.getValue("faces").tunable)
        assertEquals(ChipVisual.ERROR, byId.getValue("pose").visual)
        assertEquals("OOM", byId.getValue("pose").errorMessage)
        assertNull(byId.getValue("faces").errorMessage)
    }

    @Test
    fun action_row_lists_actions_of_enabled_pipelines_only() {
        assertTrue(ActionEntries.of(hello.pipelines, PipelineSettings()).isEmpty())
        val faces = ActionEntries.of(hello.pipelines, PipelineSettings().withEnabled("faces", true))
        assertEquals(listOf("faces/enroll", "faces/list"), faces.map { it.key })
        assertEquals("Learn face", faces.first().action.label)
    }

    @Test
    fun action_hints_cover_learn_face_and_default_others() {
        val enroll = ActionHints.of("faces", "enroll")
        assertTrue(enroll.keepOpen)
        assertTrue(enroll.offerFrontCamera)
        assertEquals(R.string.learn_face_capture, enroll.primaryLabel)
        val other = ActionHints.of("chess", "solve")
        assertFalse(other.keepOpen)
        assertFalse(other.offerFrontCamera)
        assertNull(other.primaryLabel)
    }

    @Test
    fun status_pill_per_connection_state() {
        assertEquals(StatusPillModel("Live · 9.8 fps · 48 ms", Tone.GOOD), StatusText.pill(ConnectionState.CONNECTED, false, stats))
        assertEquals(Tone.IDLE, StatusText.pill(ConnectionState.CONNECTED, true, stats).tone)
        assertEquals("Paused", StatusText.pill(ConnectionState.CONNECTED, true, stats).text)
        assertEquals(Tone.BUSY, StatusText.pill(ConnectionState.CONNECTING, false, stats).tone)
        assertEquals(StatusPillModel("Offline", Tone.BAD), StatusText.pill(ConnectionState.DISCONNECTED, false, stats))
        val noRtt = stats.copy(avgRttMs = null)
        assertEquals("Live · 9.8 fps · – ms", StatusText.pill(ConnectionState.CONNECTED, false, noRtt).text)
    }

    @Test
    fun banner_replaces_chips_until_there_is_something_to_choose() {
        assertEquals(ChipBanner.CONNECTING, StatusText.banner(ConnectionState.CONNECTING, null))
        assertEquals(ChipBanner.OFFLINE, StatusText.banner(ConnectionState.DISCONNECTED, hello.pipelines))
        assertEquals(ChipBanner.WAITING_FOR_HELLO, StatusText.banner(ConnectionState.CONNECTED, null))
        assertEquals(ChipBanner.NO_PIPELINES, StatusText.banner(ConnectionState.CONNECTED, emptyList()))
        assertNull(StatusText.banner(ConnectionState.CONNECTED, hello.pipelines))
    }

    @Test
    fun hud_rows_carry_everything_the_old_hud_showed() {
        val rows = StatusText.hudRows(ConnectionState.CONNECTED, "ws://x", "v1 · RTX 3090", false, stats)
        val byLabel = rows.associate { it.label.trim() to it.value }
        assertEquals("connected", byLabel["State"])
        assertNull(byLabel["Detail"]) // the URL is noise while connected
        assertEquals("v1 · RTX 3090", byLabel["Server"])
        assertEquals("10.0 / 9.8 fps", byLabel["Send / results"])
        assertEquals("48 ms", byLabel["Round trip"])
        assertEquals("14.3 ms", byLabel["Server time"])
        assertEquals("8.1 ms", byLabel["objects"])
        assertEquals("55 ms", byLabel["text"])
        assertEquals("1 / 2", byLabel["Drops / errors"])
        assertEquals("4", byLabel["Tracks"])

        val offline = StatusText.hudRows(ConnectionState.DISCONNECTED, "refused · retry 2.0s", null, false, stats)
        assertEquals("refused · retry 2.0s", offline.first { it.label == "Detail" }.value)
    }

    @Test
    fun server_info_shortens_the_device_name() {
        assertEquals("v0.4.0.3 · RTX 3090", StatusText.serverInfo("0.4.0.3", "cuda:0 (NVIDIA GeForce RTX 3090)"))
        assertEquals("v1 · cpu", StatusText.serverInfo("1", "cpu"))
        assertEquals("v?", StatusText.serverInfo("", ""))
    }
}
