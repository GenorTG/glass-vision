package pl.genor.glassvision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Shader
import android.graphics.Typeface
import app.cash.paparazzi.Paparazzi
import org.junit.Rule
import org.junit.Test

/** The Read text sheet in each state, and the camera screen with the Read button. */
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

    private fun box(l: Float, t: Float, r: Float, b: Float) = RectF(l, t, r, b)

    /** A station sign (2 lines) and a departure board (2 lines), as the server returns them. */
    private val reading = TextReading(
        width = 3000, height = 4000,
        lines = listOf(
            ReadLine("Ausgang", 0.96f, box(0.22f, 0.20f, 0.62f, 0.27f), 0, "de"),
            ReadLine("Notausgang 2", 0.91f, box(0.22f, 0.29f, 0.78f, 0.34f), 0, "de"),
            ReadLine("Gleis 3", 0.93f, box(0.22f, 0.58f, 0.52f, 0.63f), 1, "de"),
            ReadLine("Abfahrt 12:45", 0.88f, box(0.22f, 0.65f, 0.72f, 0.70f), 1, "de"),
        ),
        translations = mapOf(
            "pl" to listOf("Wyjście", "Wyjście awaryjne 2", "Peron 3", "Odjazd 12:45"),
            "en" to listOf("Exit", "Emergency exit 2", "Platform 3", "Departure 12:45"),
        ),
        script = "latin",
    )

    /** A dark 3:4 "photo" with the two signs drawn where [reading]'s boxes are. */
    private fun still(): Bitmap {
        val w = 600
        val h = 800
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        val bg = Paint().apply {
            shader = LinearGradient(0f, 0f, w.toFloat(), h.toFloat(), Color.rgb(0x3A, 0x44, 0x4E), Color.rgb(0x14, 0x18, 0x1C), Shader.TileMode.CLAMP)
        }
        c.drawRect(0f, 0f, w.toFloat(), h.toFloat(), bg)
        val sign = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0x1B, 0x7F, 0x3B) }
        c.drawRoundRect(0.18f * w, 0.17f * h, 0.84f * w, 0.37f * h, 12f, 12f, sign)
        val board = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(0x10, 0x2A, 0x5C) }
        c.drawRoundRect(0.18f * w, 0.55f * h, 0.80f * w, 0.73f * h, 12f, 12f, board)
        val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; typeface = Typeface.DEFAULT_BOLD }
        for (line in reading.lines) {
            text.textSize = (line.box.bottom - line.box.top) * h * 0.85f
            c.drawText(line.text, line.box.left * w, line.box.bottom * h - 0.008f * h, text)
        }
        return bmp
    }

    private fun sheet(): ReadTextSheet = ReadTextSheet(context, NoCallbacks).apply {
        languages = this@ReadTextSnapshotTest.languages
        setStill(still(), highRes = true)
    }

    @Test
    fun results_two_blocks_two_languages() {
        val s = sheet()
        s.showResult(reading, listOf("pl", "en"))
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, s.sheet.view))
    }

    /** A language being added (pending) and one the server couldn't translate into. */
    @Test
    fun results_translating_and_failed() {
        val s = sheet()
        s.showResult(reading.copy(translations = mapOf("pl" to reading.translations.getValue("pl")), errors = mapOf("ja" to "model not installed")), listOf("pl", "ja"))
        s.setTranslation("en", ReadTextSheet.Translation.Pending)
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, s.sheet.view))
    }

    @Test
    fun reading() {
        val s = sheet()
        s.showReading()
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, s.sheet.view))
    }

    @Test
    fun no_text_found() {
        val s = sheet()
        s.showResult(reading.copy(lines = emptyList(), translations = mapOf("pl" to emptyList())), listOf("pl"))
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, s.sheet.view))
    }

    @Test
    fun error_with_retry() {
        val s = sheet()
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
