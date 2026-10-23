package pl.genor.glassvision

import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Read text's pure parts: button availability, blocks, copy/share text, remembered target languages. */
class ReadTextTest {

    private fun line(text: String, block: Int, lang: String? = "de") = ReadLine(text, 0.9f, RectF(), block, lang)

    /** A sign (2 lines) and a timetable line, read in reading order. */
    private val reading = TextReading(
        width = 4000, height = 3000,
        lines = listOf(line("Ausgang", 0), line("Notausgang 2", 0), line("Gleis 3", 1), line("12:45", 1, null)),
        translations = mapOf("pl" to listOf("Wyjście", "Wyjście awaryjne 2", "Peron 3", null)),
    )

    @Test
    fun blocks_group_lines_in_reading_order() {
        assertEquals(listOf(ReadBlock(0, listOf(0, 1)), ReadBlock(1, listOf(2, 3))), ReadBlocks.of(reading.lines))
        // Interleaved ids (a sign beside a paragraph): each block keeps its lines, ordered by first appearance.
        val mixed = listOf(line("a", 5), line("b", 2), line("c", 5))
        assertEquals(listOf(ReadBlock(5, listOf(0, 2)), ReadBlock(2, listOf(1))), ReadBlocks.of(mixed))
        assertEquals(emptyList<ReadBlock>(), ReadBlocks.of(emptyList()))
    }

    @Test
    fun copy_text() {
        assertEquals("Ausgang\nNotausgang 2\n\nGleis 3\n12:45", ReadTextFormat.original(reading))
        // Untranslated lines (digits only…) keep the original.
        assertEquals(
            "Wyjście\nWyjście awaryjne 2\n\nPeron 3\n12:45",
            ReadTextFormat.translated(reading, reading.translations.getValue("pl")),
        )
    }

    @Test
    fun share_text_skips_unchanged_translations() {
        val share = ReadTextFormat.share(
            reading,
            listOf("Polish" to reading.translations.getValue("pl"), "German" to listOf(null, null, null, null)),
        )
        assertEquals(
            "Ausgang\nNotausgang 2\n\nGleis 3\n12:45\n\n— Polish —\nWyjście\nWyjście awaryjne 2\n\nPeron 3\n12:45",
            share,
        )
        assertTrue(ReadTextFormat.isUnchanged(listOf(null, null)))
        assertFalse(ReadTextFormat.isUnchanged(listOf(null, "x")))
    }

    @Test
    fun source_language_is_the_most_common() {
        assertEquals("de", ReadTextFormat.sourceLanguage(reading))
        assertNull(ReadTextFormat.sourceLanguage(reading.copy(lines = listOf(line("1", 0, null)))))
    }

    @Test
    fun read_button_needs_text_on_a_reader_and_a_connection() {
        val text = PipelineSpec(id = "text", name = "Text (OCR)", description = "", defaultEnabled = false)
        val objects = PipelineSpec(id = "objects", name = "Objects", description = "", defaultEnabled = true)
        val specs = listOf(objects, text)
        val on = PipelineSettings().withEnabled("text", true)
        val connected = ConnectionState.CONNECTED
        assertTrue(ReadAvailability.canRead(connected, specs, on, hasReader = true))
        assertFalse(ReadAvailability.canRead(connected, specs, PipelineSettings(), hasReader = true)) // text off by default
        assertFalse(ReadAvailability.canRead(connected, specs, on, hasReader = false)) // older server
        assertFalse(ReadAvailability.canRead(ConnectionState.DISCONNECTED, specs, on, hasReader = true))
        assertFalse(ReadAvailability.canRead(connected, null, on, hasReader = true)) // before hello
        assertFalse(ReadAvailability.canRead(connected, listOf(objects), on, hasReader = true))
        assertTrue(ReadAvailability.canRead(connected, listOf(text.copy(defaultEnabled = true)), PipelineSettings(), true))
    }

    // Remembered targets

    private class Prefs(var value: String? = null) {
        val targets = ReadTargets({ value }, { value = it })
    }

    private val offered = listOf(TranslationLanguage("de", "German"), TranslationLanguage("en", "English"), TranslationLanguage("pl", "Polish"))

    @Test
    fun first_read_translates_into_the_phone_language() {
        val prefs = Prefs()
        assertNull(prefs.targets.stored())
        assertEquals(listOf("pl"), prefs.targets.forRead(offered, phoneLanguage = "pl"))
        // A phone language the reader doesn't offer: nothing.
        assertEquals(emptyList<String>(), prefs.targets.forRead(offered, phoneLanguage = "kl"))
        // Languages not loaded: trust the choice.
        assertEquals(listOf("kl"), prefs.targets.forRead(emptyList(), phoneLanguage = "kl"))
    }

    @Test
    fun chosen_targets_are_remembered_in_order() {
        val prefs = Prefs()
        prefs.targets.add("en")
        prefs.targets.add("de")
        prefs.targets.add("en")
        assertEquals("en,de", prefs.value)
        assertEquals(listOf("en", "de"), prefs.targets.forRead(offered, phoneLanguage = "pl"))
        prefs.targets.remove("en")
        prefs.targets.remove("de")
        // An emptied choice stays empty (the phone language is only the first-run default).
        assertEquals("", prefs.value)
        assertEquals(emptyList<String>(), prefs.targets.forRead(offered, phoneLanguage = "pl"))
    }

    @Test
    fun stored_value_is_parsed_leniently() {
        assertEquals(listOf("pl", "de"), ReadTargets.decode(" pl, ,de,pl "))
        assertNull(ReadTargets.decode(null))
        assertEquals(emptyList<String>(), ReadTargets.decode(""))
        assertEquals("pl,de", ReadTargets.encode(listOf("pl", " de", "", "pl")))
        // Codes a server no longer offers are dropped for this read.
        assertEquals(listOf("de"), ReadTargets.initial(listOf("xx", "de"), "pl", offered))
    }

    @Test
    fun language_names_prefer_the_readers() {
        assertEquals("Polish", languageName("pl", offered))
        assertEquals("zz", languageName("zz", offered))
    }

    /**
     * v3.2 servers default the text pipeline's `live` option to "Mark text areas" for clients that send it: this app
     * sends every option hello declares (default included) in its first config, so it gets the new behaviour while
     * older apps (which never send `live`) keep reading live.
     */
    @Test
    fun live_option_is_sent_explicitly_with_its_default() {
        val live = OptionSpec(
            "live", "Live view", OptionType.CHOICE, OptionValue.Str("Mark text areas"),
            choices = listOf("Mark text areas", "Read live"),
        )
        val text = PipelineSpec(id = "text", name = "Text (OCR)", options = listOf(live))
        val first = PipelineSettings().withEnabled("text", true).resolve(listOf(text))
        val config = org.json.JSONObject(Protocol.configJson(pipelines = PipelineSettings.delta(null, first)))
        val sent = config.getJSONObject("pipelines").getJSONObject("text").getJSONObject("options")
        assertEquals("Mark text areas", sent.getString("live"))

        // The user's choice wins and goes out as a delta.
        val readLive = PipelineSettings().withEnabled("text", true).withOption("text", "live", OptionValue.Str("Read live"))
        val delta = PipelineSettings.delta(first, readLive.resolve(listOf(text)))
        assertEquals(mapOf("live" to OptionValue.Str("Read live")), delta.getValue("text").options)
    }
}
