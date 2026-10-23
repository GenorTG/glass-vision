package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PipelineSettingsTest {
    private val specs = (Protocol.parse(ProtocolTest.HELLO_V3) as ServerMessage.Hello).pipelines
    private val faces = specs.first { it.id == "faces" }
    private val threshold = faces.options.single()

    @Test
    fun untouched_pipelines_follow_server_defaults() {
        val full = PipelineSettings().resolve(specs)
        assertEquals(listOf("objects", "open_vocab", "faces", "gestures", "text"), full.keys.toList())
        assertEquals(true, full.getValue("objects").enabled)
        assertEquals(false, full.getValue("open_vocab").enabled)
        assertEquals(mapOf("prompt" to OptionValue.Str("person, cup")), full.getValue("open_vocab").options)
        assertEquals(mapOf("threshold" to OptionValue.Num(0.4)), full.getValue("faces").options)
        // Unknown-type option ("rgb") is never sent.
        assertEquals(setOf("lang"), full.getValue("gestures").options.keys)
    }

    @Test
    fun user_overrides_win_and_untouched_values_keep_defaults() {
        val s = PipelineSettings()
            .withEnabled("objects", false)
            .withEnabled("faces", true)
            .withOption("open_vocab", "prompt", OptionValue.Str("red mug"))
        val full = s.resolve(specs)
        assertEquals(false, full.getValue("objects").enabled)
        assertEquals(true, full.getValue("faces").enabled)
        assertEquals(OptionValue.Num(0.4), full.getValue("faces").options["threshold"]) // untouched → default
        assertEquals(false, full.getValue("open_vocab").enabled) // only the option was touched
        assertEquals(OptionValue.Str("red mug"), full.getValue("open_vocab").options["prompt"])
    }

    @Test
    fun stale_or_invalid_overrides_are_coerced_or_dropped() {
        val s = PipelineSettings()
            .withOption("faces", "threshold", OptionValue.Num(5.0)) // above max → clamped
            .withOption("faces", "gone", OptionValue.Bool(true)) // option no longer exists → not sent
            .withOption("gestures", "lang", OptionValue.Str("klingon")) // not a choice → default
            .withOption("text", "show_raw", OptionValue.Str("yes")) // wrong type → default
            .withEnabled("removed_pipeline", true) // unknown id → not sent
        val full = s.resolve(specs)
        assertEquals(OptionValue.Num(0.8), full.getValue("faces").options["threshold"])
        assertEquals(setOf("threshold"), full.getValue("faces").options.keys)
        assertEquals(OptionValue.Str("latin"), full.getValue("gestures").options["lang"])
        assertEquals(OptionValue.Bool(true), full.getValue("text").options["show_raw"])
        assertFalse("removed_pipeline" in full)
        assertEquals(OptionValue.Num(0.8), s.optionValue(faces, threshold))
    }

    @Test
    fun json_blob_round_trips_and_garbage_yields_defaults() {
        val s = PipelineSettings()
            .withEnabled("faces", true)
            .withOption("faces", "threshold", OptionValue.Num(0.55))
            .withOption("open_vocab", "prompt", OptionValue.Str("keys, \"glasses\""))
            .withOption("text", "show_raw", OptionValue.Bool(false))
            .withOption("x", "count", OptionValue.Num(3.0))
        assertEquals(s, PipelineSettings.fromJson(s.toJson()))
        assertEquals(PipelineSettings(), PipelineSettings.fromJson("{not json"))
        assertEquals(PipelineSettings(), PipelineSettings.fromJson(null))
        assertNull(PipelineSettings.fromJson("""{"faces":{"options":{"a":1}}}""").overrides.getValue("faces").enabled)
    }

    @Test
    fun delta_sends_everything_first_then_only_changes() {
        val before = PipelineSettings().resolve(specs)
        assertEquals(before, PipelineSettings.delta(null, before))
        assertTrue(PipelineSettings.delta(before, before).isEmpty())

        val after = PipelineSettings()
            .withEnabled("faces", true)
            .withOption("open_vocab", "prompt", OptionValue.Str("keys"))
            .resolve(specs)
        assertEquals(
            mapOf(
                "open_vocab" to PipelineConfig(null, mapOf("prompt" to OptionValue.Str("keys"))),
                "faces" to PipelineConfig(enabled = true),
            ),
            PipelineSettings.delta(before, after),
        )
        // A pipeline the previous config didn't have goes out in full.
        val grown = after + ("new" to PipelineConfig(false, mapOf("k" to OptionValue.Bool(true))))
        assertEquals(mapOf("new" to grown.getValue("new")), PipelineSettings.delta(after, grown))
    }

    @Test
    fun number_steps_map_slider_positions() {
        val steps = NumberSteps(0.2, 0.8, 0.05)
        assertEquals(12, steps.count)
        assertEquals(4, steps.toProgress(0.4))
        assertEquals(0.4, steps.fromProgress(4), 1e-12)
        assertEquals(0.8, steps.fromProgress(99), 1e-12)
        assertEquals("0.40", steps.format(0.4))
        val ints = NumberSteps(1.0, 10.0, 1.0)
        assertEquals(9, ints.count)
        assertEquals("3", ints.format(3.0))
        assertEquals(100, NumberSteps(0.0, 1.0, null).count)
        assertEquals("3", NumberSteps.plain(3.0))
        assertEquals("0.25", NumberSteps.plain(0.25))
    }

    @Test
    fun catalog_tracks_hello_status_and_disconnect() {
        val catalog = PipelineCatalog()
        val changes = mutableListOf<PipelineCatalog.Change>()
        catalog.addListener { changes += it }
        catalog.onStatus(ServerMessage.PipelineStatusChange("faces", PipelineStatus.READY, null)) // before hello: ignored
        catalog.onHello(Protocol.parse(ProtocolTest.HELLO_V3) as ServerMessage.Hello)
        catalog.onStatus(ServerMessage.PipelineStatusChange("faces", PipelineStatus.ERROR, "OOM"))
        assertEquals(PipelineStatus.ERROR, catalog.find("faces")!!.status)
        assertEquals("OOM", catalog.find("faces")!!.statusMessage)
        catalog.clear()
        assertNull(catalog.specs)
        assertEquals(
            listOf(PipelineCatalog.Change.Rebuilt, PipelineCatalog.Change.Status("faces"), PipelineCatalog.Change.Rebuilt),
            changes,
        )
    }
}
