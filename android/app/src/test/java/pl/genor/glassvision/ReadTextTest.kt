package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Read text's pure parts: button availability, blocks, copy/share text, remembered target languages. */
class ReadTextTest {

    private fun line(text: String, block: Int, lang: String? = "de", box: TextBox = TextBox(0.1f, 0.1f, 0.5f, 0.2f)) =
        ReadLine(text, 0.9f, box, block, lang)

    /** A sign (2 lines) and a timetable line, read in reading order (a v3.2 answer: blocks from the lines). */
    private val lines = listOf(line("Ausgang", 0), line("Notausgang 2", 0), line("Gleis 3", 1), line("12:45", 1, null))
    private val reading = TextReading(
        width = 4000, height = 3000, lines = lines,
        blocks = ReadBlocks.fromLines(lines, mapOf("pl" to listOf("Wyjście", "Wyjście awaryjne 2", "Peron 3", null))),
    )

    @Test
    fun v32_lines_become_blocks_with_joined_translations() {
        val blocks = reading.blocks
        assertEquals(listOf(listOf(0, 1), listOf(2, 3)), blocks.map { it.lines })
        assertEquals(listOf(0, 1), blocks.map { it.id })
        assertEquals("Ausgang\nNotausgang 2", blocks[0].text)
        assertEquals(listOf(BlockKind.LIST, BlockKind.LIST), blocks.map { it.kind })
        assertEquals(listOf("de", "de"), blocks.map { it.lang })
        // Untranslated lines (digits only…) keep the original inside a translated block.
        assertEquals(listOf("Wyjście\nWyjście awaryjne 2", "Peron 3\n12:45"), reading.translation("pl"))
        assertEquals(setOf("pl"), reading.translated)
        // Interleaved ids (a sign beside a paragraph): each block keeps its lines, ordered by first appearance.
        val mixed = ReadBlocks.fromLines(listOf(line("a", 5), line("b", 2), line("c", 5)))
        assertEquals(listOf(5 to listOf(0, 2), 2 to listOf(1)), mixed.map { it.id to it.lines })
        assertEquals(emptyList<ReadBlock>(), ReadBlocks.fromLines(emptyList()))
    }

    @Test
    fun v32_block_box_is_the_union_and_prose_is_reflowed() {
        val prose = listOf(
            line("Bitte halten Sie diese", 0, box = TextBox(0.1f, 0.10f, 0.6f, 0.15f)),
            line("Tür immer geschlossen", 0, box = TextBox(0.1f, 0.16f, 0.58f, 0.21f)),
        )
        val b = ReadBlocks.fromLines(prose, mapOf("en" to listOf("Please keep this", "door closed at all times"), "fr" to listOf(null, null))).single()
        assertEquals(TextBox(0.1f, 0.10f, 0.6f, 0.21f), b.box)
        assertEquals(BlockKind.PROSE, b.kind)
        assertEquals("Bitte halten Sie diese Tür immer geschlossen", b.display)
        assertEquals("Please keep this door closed at all times", b.translations["en"])
        // A target that translated none of the block's lines: present, null.
        assertTrue("fr" in b.translations)
        assertNull(b.translations["fr"])
    }

    @Test
    fun copy_text() {
        assertEquals("Ausgang\nNotausgang 2\n\nGleis 3\n12:45", ReadTextFormat.original(reading))
        assertEquals(
            "Wyjście\nWyjście awaryjne 2\n\nPeron 3\n12:45",
            ReadTextFormat.translated(reading, reading.translation("pl")),
        )
        // Untranslated blocks keep the original.
        assertEquals("Exit\n\nGleis 3\n12:45", ReadTextFormat.translated(reading, listOf("Exit", null)))
        assertEquals(
            "Ausgang\nNotausgang 2\n\n— Polish —\nWyjście\nWyjście awaryjne 2",
            ReadTextFormat.block(reading.blocks[0], listOf("Polish" to "Wyjście\nWyjście awaryjne 2", "German" to null)),
        )
    }

    @Test
    fun share_text_skips_unchanged_translations() {
        val share = ReadTextFormat.share(
            reading,
            listOf("Polish" to reading.translation("pl"), "German" to listOf(null, null)),
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
        val digits = listOf(line("1", 0, null))
        assertNull(ReadTextFormat.sourceLanguage(reading.copy(lines = digits, blocks = ReadBlocks.fromLines(digits))))
    }

    /** The interface's default block translation (readers without one): line by line, recombined per block. */
    @Test
    fun default_block_translation_goes_line_by_line() {
        val calls = mutableListOf<Triple<List<String>, List<String?>, String>>()
        val reader = object : TextReader {
            override fun read(still: android.graphics.Bitmap, translateTo: List<String>) = throw UnsupportedOperationException()
            override fun languages() = emptyList<TranslationLanguage>()
            override fun translate(lines: List<String>, from: List<String?>, to: String): List<String?> {
                calls += Triple(lines, from, to)
                return lines.map { mapOf("Ausgang" to "Exit", "Gleis 3" to "Platform 3")[it] }
            }
        }
        val blocks = reading.blocks + ReadBlock(9, emptyList(), text = "42", lang = null)
        assertEquals(listOf("Exit\nNotausgang 2", "Platform 3\n12:45", null), reader.translateBlocks(blocks, "en"))
        assertEquals(Triple(listOf("Ausgang", "Notausgang 2", "Gleis 3", "12:45", "42"), listOf("de", "de", "de", "de", null), "en"), calls.single())
    }

    @Test
    fun the_view_choice_is_remembered() {
        var stored: String? = null
        val views = ReadViewChoice({ stored }, { stored = it })
        assertEquals(ReadView.CARDS, views.get())
        views.set(ReadView.OVERLAY)
        assertEquals("OVERLAY", stored)
        assertEquals(ReadView.OVERLAY, views.get())
        stored = "SOMETHING_OLD"
        assertEquals(ReadView.CARDS, views.get())
    }

    @Test
    fun calendar_dates_are_local_midnight() {
        val utc = java.util.TimeZone.getTimeZone("UTC")
        assertEquals(1_792_281_600_000L, ReadActions.dateMillis("2026-10-18", utc))
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
