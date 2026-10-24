package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** On-device text's pure logic: reading order, blocks, crop math, plans per OCR language, languages. */
class TextLayoutTest {
    private fun line(text: String, l: Float, t: Float, r: Float, b: Float, block: Int, conf: Float = 0.9f) =
        OcrLine(text, conf, TextBox(l, t, r, b), block)

    @Test
    fun rows_go_top_to_bottom_then_across() {
        val boxes = listOf(TextBox(300f, 12f, 400f, 40f), TextBox(0f, 100f, 100f, 130f), TextBox(0f, 10f, 100f, 38f))
        assertEquals(listOf(2, 0, 1), TextLayout.rowOrder(boxes.indices.toList()) { boxes[it] })
        assertEquals(listOf(0, 2, 1), TextLayout.rowOrder(boxes.indices.toList(), rtl = true) { boxes[it] })
        assertTrue(TextLayout.sameRow(TextBox(0f, 10f, 1f, 38f), TextBox(0f, 12f, 1f, 40f)))
        assertFalse(TextLayout.sameRow(TextBox(0f, 10f, 1f, 38f), TextBox(0f, 30f, 1f, 58f)))
    }

    @Test
    fun quads_measure_along_and_across_the_text() {
        val q = Quad(listOf(Pt(0f, 0f), Pt(80f, 60f), Pt(68f, 76f), Pt(-12f, 16f))) // 100 long, 20 thick, 36.87° clockwise
        assertEquals(36.87f, q.angle, 0.01f)
        assertEquals(100f, q.length, 0.01f)
        assertEquals(20f, q.thickness, 0.01f)
        assertEquals(TextBox(-12f, 0f, 80f, 76f), q.bounds())
        assertEquals(Quad.of(TextBox(1f, 2f, 3f, 4f)).bounds(), TextBox(1f, 2f, 3f, 4f))
        assertEquals(0f, Quad.of(TextBox(1f, 2f, 3f, 4f)).angle)
    }

    @Test
    fun the_engines_language_tag_counts_when_language_id_has_no_answer() {
        val lines = listOf(OcrLine("Zahnarztpraxis", 0.9f, TextBox(0f, 0f, 1f, 1f), 0, lang = "de"), OcrLine("Weiter so", 0.9f, TextBox(0f, 0f, 1f, 1f), 0, lang = "und"))
        assertEquals(listOf("de", "de"), TextLanguages.assign(lines) { null })
    }

    @Test
    fun crops_are_padded_clipped_and_scaled_for_tesseract() {
        assertEquals(CropRect(90, 90, 210, 150), TextLayout.crop(TextBox(100f, 100f, 200f, 140f), 1000, 1000))
        assertEquals(CropRect(0, 0, 54, 23), TextLayout.crop(TextBox(2f, 3f, 50f, 19f), 54, 30)) // clipped to the image
        assertNull(TextLayout.crop(TextBox(10f, 10f, 11f, 11f), 1000, 1000))
        assertNull(TextLayout.crop(TextBox(995f, 10f, 1005f, 30f), 990, 1000)) // outside the image
        assertEquals(4f, TextLayout.cropScale(6f)) // tiny text: at most 4×
        assertEquals(2f, TextLayout.cropScale(24f))
        assertEquals(1f, TextLayout.cropScale(48f))
        assertEquals(0.5f, TextLayout.cropScale(400f)) // huge text: at least 0.5×
        assertEquals(1f, TextLayout.cropScale(0f))
        assertEquals(240 to 80, TextLayout.scaled(120, 40, 2f))
        assertEquals(1 to 1, TextLayout.scaled(1, 1, 0.5f))
    }

    @Test
    fun stills_fit_the_long_side() {
        assertEquals(2560 to 1920, TextLayout.fitLongSide(4000, 3000))
        assertEquals(1440 to 2560, TextLayout.fitLongSide(3024, 5376))
        assertEquals(810 to 1080, TextLayout.fitLongSide(810, 1080))
        assertEquals(1600 to 900, TextLayout.fitLongSide(1920, 1080, 1600))
    }

    @Test
    fun covered_is_the_share_of_the_first_box() {
        assertEquals(1f, TextLayout.covered(TextBox(10f, 10f, 20f, 20f), TextBox(0f, 0f, 100f, 100f)))
        assertEquals(0.5f, TextLayout.covered(TextBox(0f, 0f, 10f, 10f), TextBox(5f, 0f, 100f, 100f)))
        assertEquals(0f, TextLayout.covered(TextBox(0f, 0f, 10f, 10f), TextBox(20f, 0f, 30f, 10f)))
    }

    @Test
    fun every_ocr_language_has_a_reader_and_a_script() {
        val packOnly = OcrLanguage.entries.filter { it.script == TextScript.OTHER }.map { it.code }.toSet()
        assertEquals(setOf("ara", "rus", "ukr", "ell", "heb", "tha", "ben", "tam", "bul", "srp"), packOnly)
        assertEquals(TextScript.CHINESE, OcrLanguage.CHINESE_TRADITIONAL.script)
        assertEquals(TextScript.DEVANAGARI, OcrLanguage.HINDI.script)
        assertEquals(TextScript.LATIN, OcrLanguage.VIETNAMESE.script)
        assertEquals(TextScript.LATIN, OcrLanguage.BOSNIAN.script)
        // Latin-script languages read Latin letters; the rest don't.
        for (lang in OcrLanguage.entries) {
            assertEquals(lang.code, lang.script == TextScript.LATIN, Character.UnicodeScript.LATIN in lang.letters)
        }
        assertFalse(OcrLanguage.ARABIC.latinFindsLines)
        assertTrue(OcrLanguage.RUSSIAN.latinFindsLines)
    }

    @Test
    fun plans_follow_the_ocr_language_choice() {
        assertEquals(OcrPlan(TextScript.LATIN), OcrPlan.forChoice(OcrLanguage.POLISH.choice, listOf(OcrLanguage.RUSSIAN), listOf("ru")))
        assertEquals(
            OcrPlan(TextScript.LATIN, packs = listOf(OcrLanguage.RUSSIAN)),
            OcrPlan.forChoice(OcrLanguage.RUSSIAN.choice, emptyList(), listOf("en")),
        )
        assertEquals(
            OcrPlan(TextScript.JAPANESE, fallback = listOf(OcrLanguage.JAPANESE)),
            OcrPlan.forChoice(OcrLanguage.JAPANESE.choice, emptyList(), listOf("en")),
        )
        // Unknown choice (an old pref): English → Latin.
        assertEquals(OcrPlan(TextScript.LATIN), OcrPlan.forChoice("Klingon · tlh", emptyList(), emptyList()))
    }

    @Test
    fun auto_uses_phone_languages_and_installed_packs_without_new_downloads() {
        val auto = OcrLanguage.AUTO_CHOICE
        assertEquals(OcrPlan(TextScript.LATIN, auto = true), OcrPlan.forChoice(auto, listOf(OcrLanguage.POLISH), listOf("pl", "en")))
        val ukrainian = OcrPlan.forChoice(auto, listOf(OcrLanguage.GREEK, OcrLanguage.GERMAN), listOf("uk", "en"))
        assertEquals(TextScript.LATIN, ukrainian.script)
        assertEquals(listOf(OcrLanguage.UKRAINIAN, OcrLanguage.GREEK), ukrainian.packs)
        assertEquals(OcrPlan.RETRY_BELOW, ukrainian.retryBelow)
        assertTrue(ukrainian.auto)
        val japanese = OcrPlan.forChoice(auto, emptyList(), listOf("ja"))
        assertEquals(TextScript.JAPANESE to listOf(OcrLanguage.JAPANESE), japanese.script to japanese.fallback)
        val many = OcrPlan.forChoice(auto, OcrLanguage.entries.filter { it.script == TextScript.OTHER }, listOf("he"))
        assertEquals(OcrPlan.AUTO_MAX_PACKS, many.packs.size)
        assertEquals(OcrLanguage.HEBREW, many.packs.first())
    }

    @Test
    fun a_pack_reading_wins_only_in_its_script_and_more_confident() {
        val rus = OcrLanguage.RUSSIAN
        // Measured on the emulator: ML Kit Latin vs the rus pack on the same crops.
        assertEquals("ВЫХОД", TextLanguages.choose(Reading("BbIXOI", 0.55f), listOf(rus to Reading("ВЫХОД", 0.81f))).text)
        assertEquals("EXIT 12", TextLanguages.choose(Reading("EXIT 12", 0.62f), listOf(rus to Reading("EXIT 12", 0.95f))).text)
        // Latin read as Cyrillic look-alikes is in the pack's script; ML Kit's higher confidence keeps it.
        assertEquals("EXIT 12", TextLanguages.choose(Reading("EXIT 12", 0.83f), listOf(rus to Reading("ЕХТ 12", 0.62f))).text)
        // Too unsure, or empty: ML Kit's stays.
        assertEquals("Cero", TextLanguages.choose(Reading("Cero", 0.2f), listOf(rus to Reading("сео", 0.25f))).text)
        assertEquals("x", TextLanguages.choose(Reading("x", 0.1f), listOf(rus to Reading("", 0.9f))).text)
        // The most confident valid one of several.
        val ell = OcrLanguage.GREEK
        assertEquals(
            Reading("ΕΞΟΔΟΣ", 0.96f),
            TextLanguages.choose(Reading("EEOAOE", 0.4f), listOf(rus to Reading("ЕЕОДОЕ", 0.7f), ell to Reading("ΕΞΟΔΟΣ", 0.96f))),
        )
    }

    @Test
    fun in_script_counts_letters_only() {
        assertTrue(TextLanguages.inScript("ВЫХОД 12", OcrLanguage.RUSSIAN))
        assertFalse(TextLanguages.inScript("EXIT 12", OcrLanguage.RUSSIAN))
        assertTrue(TextLanguages.inScript("Ab ВЫ", OcrLanguage.RUSSIAN)) // half
        assertFalse(TextLanguages.inScript("12 34", OcrLanguage.RUSSIAN))
        assertTrue(TextLanguages.inScript("出口", OcrLanguage.CHINESE_SIMPLIFIED))
        assertTrue(TextLanguages.inScript("でぐち", OcrLanguage.JAPANESE))
        assertTrue(TextLanguages.inScript("ทางออก", OcrLanguage.THAI)) // vowel marks aren't letters
    }

    @Test
    fun language_id_tags_become_source_codes() {
        assertEquals("de", TextLanguages.sourceCode("de"))
        assertEquals("zh", TextLanguages.sourceCode("zh-Hant"))
        assertNull(TextLanguages.sourceCode("und"))
        assertNull(TextLanguages.sourceCode("ja-Latn")) // romanized: nothing translates it
        assertNull(TextLanguages.sourceCode(null))
        assertNull(TextLanguages.sourceCode(""))
    }

    @Test
    fun short_lines_inherit_their_blocks_language() {
        val ids = mutableListOf<String>()
        val identify = { text: String ->
            ids += text
            when {
                "Notausgang" in text || "freihalten" in text -> "de"
                "Exit" in text -> "en"
                else -> null
            }
        }
        val lines = listOf(
            line("Ausgang", 0f, 0f, 1f, 1f, 0), // 7 letters: too short alone
            line("Bitte freihalten", 0f, 0f, 1f, 1f, 0),
            line("Exit", 0f, 0f, 1f, 1f, 1),
            line("Notausgang", 0f, 0f, 1f, 1f, 1),
            line("Kasse", 0f, 0f, 1f, 1f, 2), // a block of its own, too short to identify: the still's majority
            line("42", 0f, 0f, 1f, 1f, 2), // no letters: null
        )
        assertEquals(listOf("de", "de", "de", "de", "de", null), TextLanguages.assign(lines, identify))
        assertFalse("short lines are never identified alone", "Ausgang" in ids || "Exit" in ids)
        assertEquals(listOf<String?>(null), TextLanguages.assign(listOf(line("Hi", 0f, 0f, 1f, 1f, 0)), identify))
    }

    @Test
    fun only_wordy_lines_in_another_language_are_translated() {
        assertTrue(TextLanguages.translatable("Ausgang", "de", "pl"))
        assertFalse(TextLanguages.translatable("Ausgang", "de", "de"))
        assertFalse(TextLanguages.translatable("Ausgang", null, "pl"))
        assertFalse(TextLanguages.translatable("A4 2024", "de", "pl")) // more digits than letters
        assertFalse(TextLanguages.translatable("x", "de", "pl"))
    }

    @Test
    fun translation_targets_are_named_in_english_and_sorted() {
        val targets = TextLanguages.targets(listOf("pl", "de", "he", "zh", "pl"))
        assertEquals(listOf("Chinese", "German", "Hebrew", "Polish"), targets.map { it.name })
        assertEquals(listOf("zh", "de", "he", "pl"), targets.map { it.code })
    }

    @Test
    fun the_local_text_pipeline_matches_the_read_text_contract() {
        val text = LocalCatalog.specs.first { it.id == LocalCatalog.TEXT }
        val live = text.options.first { it.key == "live" }
        assertEquals(OptionValue.Str("Mark text areas"), live.default)
        assertEquals(listOf("Mark text areas", "Read live"), live.choices)
        assertEquals(OptionValue.Str(OcrLanguage.AUTO_CHOICE), text.options.first { it.key == LocalCatalog.OPT_OCR_LANGUAGE }.default)
        assertTrue(text.description, "ML Kit" in text.description)
        assertTrue(text.intervalMs > 0) // ~200 ms per frame: stays on the slow lane
    }
}
