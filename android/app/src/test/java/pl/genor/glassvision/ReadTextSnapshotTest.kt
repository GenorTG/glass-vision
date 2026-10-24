package pl.genor.glassvision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Shader
import android.graphics.Typeface
import app.cash.paparazzi.Paparazzi
import org.junit.Rule
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

/** The Read text sheet in each state and both result views, and the camera screen with the Read button. */
class ReadTextSnapshotTest {
    @get:Rule
    val paparazzi = Paparazzi(deviceConfig = SnapshotFixtures.PHONE, theme = SnapshotFixtures.THEME)

    private val context get() = paparazzi.context

    private object NoCallbacks : ReadTextSheet.Callbacks {
        override fun onRetry() = Unit
        override fun onAddLanguage() = Unit
        override fun onRemoveLanguage(code: String) = Unit
        override fun onCopy(text: String) = Unit
        override fun onShare(text: String) = Unit
    }

    private val languages = listOf(
        TranslationLanguage("en", "English"), TranslationLanguage("de", "German"), TranslationLanguage("pl", "Polish"),
    )

    /** One sign on the still: [lines] (text, height as a share of the still's height) from ([x], [y]), tilted [deg]. */
    private class Sign(
        val x: Float, val y: Float, val deg: Float, val color: Int, val ink: Int,
        val lines: List<Pair<String, Float>>, val bold: Boolean = true,
        val kind: BlockKind = BlockKind.LIST, val translations: Map<String, String?> = emptyMap(), val suspect: Boolean = false,
    )

    private val w = 600
    private val h = 800

    private val signs = listOf(
        Sign(
            0.16f, 0.14f, -4f, Color.rgb(0x1B, 0x7F, 0x3B), Color.WHITE, listOf("Ausgang" to 0.065f, "Notausgang 2" to 0.05f),
            translations = mapOf("pl" to "Wyjście\nWyjście awaryjne 2", "en" to "Exit\nEmergency exit 2"),
        ),
        Sign(
            0.12f, 0.43f, 0f, Color.rgb(0xF2, 0xEC, 0xDC), Color.rgb(0x22, 0x22, 0x22),
            listOf("Bitte halten Sie diese" to 0.032f, "Tür immer geschlossen." to 0.032f, "Info: 030 123 4567" to 0.032f),
            bold = false, kind = BlockKind.PROSE,
            translations = mapOf(
                "pl" to "Prosimy zawsze trzymać te drzwi zamknięte. Informacja: 030 123 4567",
                "en" to "Please keep this door closed at all times. Info: 030 123 4567",
            ),
        ),
        Sign(
            0.20f, 0.68f, 3f, Color.rgb(0x10, 0x2A, 0x5C), Color.rgb(0xFF, 0xD4, 0x00), listOf("Gleis 3" to 0.05f, "Abfahrt 12:45" to 0.05f),
            translations = mapOf("pl" to "Peron 3\nOdjazd 12:45", "en" to "Platform 3\nDeparture 12:45"),
        ),
    )

    private fun paint(sign: Sign, size: Float) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = sign.ink
        typeface = if (sign.bold) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
        textSize = size
    }

    /** The signs drawn on a dark 3:4 "photo", and the reading of them (quads where the text really is). */
    private fun scene(signs: List<Sign> = this.signs, read: (Sign, Int) -> String = { s, i -> s.lines[i].first }): Pair<Bitmap, TextReading> {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        c.drawRect(0f, 0f, w.toFloat(), h.toFloat(), Paint().apply {
            shader = LinearGradient(0f, 0f, w.toFloat(), h.toFloat(), Color.rgb(0x3A, 0x44, 0x4E), Color.rgb(0x14, 0x18, 0x1C), Shader.TileMode.CLAMP)
        })
        val lines = ArrayList<ReadLine>()
        val blocks = ArrayList<ReadBlock>()
        for ((b, sign) in signs.withIndex()) {
            val a = Math.toRadians(sign.deg.toDouble())
            val ox = sign.x * w
            val oy = sign.y * h
            fun at(px: Float, py: Float) = Pt(((ox + px * cos(a) - py * sin(a)) / w).toFloat(), ((oy + px * sin(a) + py * cos(a)) / h).toFloat())
            var top = 0f
            val rows = sign.lines.map { (text, share) ->
                val size = share * h
                val width = paint(sign, size).measureText(text)
                Triple(text, top, size to width).also { top += size * 1.25f }
            }
            val bottom = top - rows.last().third.first * 0.25f
            val right = rows.maxOf { it.third.second }
            val pad = 0.35f * rows.first().third.first
            c.save()
            c.translate(ox, oy)
            c.rotate(sign.deg)
            c.drawRoundRect(-pad, -pad, right + pad, bottom + pad, 10f, 10f, Paint(Paint.ANTI_ALIAS_FLAG).apply { color = sign.color })
            val first = lines.size
            for ((i, row) in rows.withIndex()) {
                val (text, y, sw) = row
                val (size, width) = sw
                c.drawText(text, 0f, y + size * 0.8f, paint(sign, size))
                val quad = Quad(listOf(at(0f, y), at(width, y), at(width, y + size), at(0f, y + size)))
                lines += ReadLine(read(sign, i), 0.9f, quad.bounds(), b, "de", quad = quad, script = "latin", suspect = sign.suspect)
            }
            c.restore()
            val quad = Quad(listOf(at(0f, 0f), at(right, 0f), at(right, bottom), at(0f, bottom)))
            blocks += ReadBlock(
                id = b, lines = (first until lines.size).toList(), box = quad.bounds(), quad = quad,
                text = (first until lines.size).joinToString("\n") { lines[it].text }, kind = sign.kind, lang = "de",
                suspect = sign.suspect, translations = sign.translations,
            )
        }
        return bmp to TextReading(3000, 4000, lines, blocks, script = "latin")
    }

    private fun sheet(still: Bitmap, view: ReadView = ReadView.CARDS): ReadTextSheet = ReadTextSheet(context, NoCallbacks, view).apply {
        languages = this@ReadTextSnapshotTest.languages
        setStill(still, highRes = true)
    }

    @Test
    fun results_cards_three_blocks_two_languages() {
        val (still, reading) = scene()
        val s = sheet(still)
        s.showResult(reading, listOf("pl", "en"))
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, s.sheet.view))
    }

    /** "On picture": each block erased with its sampled background and the Polish text painted in, rotated with it. */
    @Test
    fun results_overlay_translated() {
        val (still, reading) = scene()
        val s = sheet(still, ReadView.OVERLAY)
        s.showResult(reading, listOf("pl", "en"))
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, s.sheet.view))
    }

    /** "On picture" with "Show original" on and the second sign tapped (highlighted on the still and its card). */
    @Test
    fun results_overlay_original_selected() {
        val (still, reading) = scene()
        val s = sheet(still, ReadView.OVERLAY)
        s.showResult(reading, listOf("pl", "en"))
        s.sheet.view.findViewById<com.google.android.material.chip.Chip>(R.id.read_show_original).isChecked = true
        s.select(1)
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, s.sheet.view))
    }

    /** A sign ML Kit Latin misread (ВЫХОД → "BbIXOI"): marked, with the (not yet available) "Read carefully". */
    @Test
    fun results_suspect_block_and_actions() {
        val signs = listOf(
            Sign(0.18f, 0.16f, 0f, Color.rgb(0x1B, 0x7F, 0x3B), Color.WHITE, listOf("ВЫХОД" to 0.07f), suspect = true,
                translations = mapOf("pl" to null)),
            Sign(0.12f, 0.45f, 0f, Color.rgb(0xF2, 0xEC, 0xDC), Color.rgb(0x22, 0x22, 0x22),
                listOf("Kontakt: +48 600 700 800" to 0.032f, "info@kino.pl · www.kino.pl" to 0.032f, "Gültig bis 24.10.2026" to 0.032f),
                bold = false, translations = mapOf("pl" to "Kontakt: +48 600 700 800\ninfo@kino.pl · www.kino.pl\nWażne do 24.10.2026")),
        )
        val (still, reading) = scene(signs) { s, i -> if (s.suspect) "BbIXOI" else s.lines[i].first }
        val s = sheet(still)
        s.showResult(reading.copy(blocks = reading.blocks.mapIndexed { i, b -> if (i == 0) b.copy(lang = null) else b }), listOf("pl"))
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, s.sheet.view))
    }

    /** A language being added (pending) and one the server couldn't translate into. */
    @Test
    fun results_translating_and_failed() {
        val (still, reading) = scene()
        val s = sheet(still)
        val polishOnly = reading.copy(blocks = reading.blocks.map { b -> b.copy(translations = b.translations.filterKeys { it == "pl" }) })
        s.showResult(polishOnly.copy(errors = mapOf("ja" to "model not installed")), listOf("pl", "ja"))
        s.setTranslation("en", ReadTextSheet.Translation.Pending)
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, s.sheet.view))
    }

    @Test
    fun reading() {
        val s = sheet(scene().first)
        s.showReading()
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, s.sheet.view))
    }

    @Test
    fun no_text_found() {
        val (still, reading) = scene()
        val s = sheet(still)
        s.showResult(reading.copy(lines = emptyList(), blocks = emptyList()), listOf("pl"))
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, s.sheet.view))
    }

    @Test
    fun error_with_retry() {
        val s = sheet(scene().first)
        s.showError("OCR model could not be loaded: CUDA out of memory (HTTP 503)")
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, s.sheet.view))
    }

    /** Text (OCR) on, server with `text_read`: the Read button above the shutter. */
    @Test
    fun camera_with_read_button() {
        paparazzi.snapshot(
            SnapshotFixtures.mainScreen(context, SnapshotFixtures.liveState().copy(canReadText = true), SnapshotFixtures.detections)
        )
    }
}
