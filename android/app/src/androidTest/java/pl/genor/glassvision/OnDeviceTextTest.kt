package pl.genor.glassvision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * On-device text on real images: ML Kit v2 on ultralytics' bus.jpg (a Spanish bus: "cero emisiones" on its side,
 * which whole-frame Tesseract read as "re"), rendered signs in Latin and in scripts only a Tesseract pack reads
 * (downloaded on first use, like the app does), reading order, translation, and live latency (logged, tag
 * OnDeviceTextTest). Needs internet on the device/emulator for packs and translation models.
 */
@RunWith(AndroidJUnit4::class)
class OnDeviceTextTest {
    companion object {
        private lateinit var context: Context
        private lateinit var models: ModelManager
        private lateinit var loader: OnDeviceLoader
        private lateinit var captureRoot: File
        private const val TAG = "OnDeviceTextTest"

        @BeforeClass
        @JvmStatic
        fun setUp() {
            context = InstrumentationRegistry.getInstrumentation().targetContext
            captureRoot = File(context.cacheDir, "captures-text-test")
            models = ModelManager(File(context.filesDir, "models"), OkHttpDownloader())
            loader = OnDeviceLoader(context, models, CaptureStore(captureRoot))
        }

        @AfterClass
        @JvmStatic
        fun tearDown() {
            captureRoot.deleteRecursively()
        }
    }

    private fun asset(name: String): Bitmap {
        val assets = InstrumentationRegistry.getInstrumentation().context.assets
        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        return assets.open(name).use { requireNotNull(BitmapFactory.decodeStream(it, null, options)) }
    }

    /** Black bold text on white, one string per line, at [x] (each list = one sign at its own x). */
    private fun sign(vararg signs: Pair<Float, List<String>>, width: Int = 1280, height: Int = 720, size: Float = 90f): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.BLACK
            textSize = size
            typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)
        }
        Canvas(bitmap).apply {
            drawColor(Color.WHITE)
            for ((x, lines) in signs) lines.forEachIndexed { i, line -> drawText(line, x, 200f + i * size * 1.4f, paint) }
        }
        return bitmap
    }

    private fun textOptions(
        language: String = OcrLanguage.AUTO_CHOICE,
        live: String = LocalCatalog.LIVE_READ,
        translateTo: String = LocalCatalog.TRANSLATE_OFF,
    ): Map<String, OptionValue> = mapOf(
        LocalCatalog.OPT_OCR_LANGUAGE to OptionValue.Str(language),
        LocalCatalog.OPT_LIVE to OptionValue.Str(live),
        LocalCatalog.OPT_TRANSLATE_TO to OptionValue.Str(translateTo),
    )

    private fun <T> reader(options: Map<String, OptionValue> = textOptions(), block: (LocalTextReader) -> T): T {
        val reader = LocalTextReader(context, models) { options }
        try {
            return block(reader)
        } finally {
            reader.close()
        }
    }

    private fun live(image: Bitmap, options: Map<String, OptionValue>): List<PixelDetection> {
        val pipeline = loader.load(LocalCatalog.TEXT, options, false)
        try {
            return pipeline.run(BitmapFrame(image), 1, options)
        } finally {
            pipeline.close()
        }
    }

    private fun squash(s: String) = s.uppercase().filter { it.isLetterOrDigit() }

    @Test
    fun read_finds_cero_emisiones_on_bus_jpg() {
        val reading = reader { it.read(asset("bus.jpg"), emptyList()) }
        Log.i(TAG, "bus.jpg via ${reading.script}: ${reading.lines.map { "${it.text} (${it.conf}, block ${it.block}, ${it.lang})" }}")
        val all = reading.lines.joinToString(" ") { it.text }.lowercase()
        assertTrue(all, "emisiones" in all)
        assertEquals(810 to 1080, reading.width to reading.height)
        assertTrue(reading.lines.all { it.box.left >= 0f && it.box.right <= 1f && it.box.top < it.box.bottom })
        assertTrue(reading.script.orEmpty(), reading.script.orEmpty().startsWith("ML Kit"))
    }

    @Test
    fun live_marks_text_areas_by_default_and_reads_on_request() {
        val bus = asset("bus.jpg")
        val marked = live(bus, textOptions(live = LocalCatalog.LIVE_MARK))
        assertTrue(marked.isNotEmpty())
        assertTrue(marked.map { it.label }.toString(), marked.all { it.label == LocalCatalog.TEXT_LABEL })
        val read = live(bus, textOptions())
        assertTrue(read.map { it.label }.toString(), read.any { "emisiones" in it.label.lowercase() })
        assertEquals(marked.size, read.size)
    }

    @Test
    fun reads_a_latin_sign_and_keeps_reading_order_per_block() {
        // Two signs side by side, two lines each: left sign first, its lines together.
        val image = sign(80f to listOf("EXIT", "GATE 42"), 720f to listOf("TOILETS", "LIFT"))
        val reading = reader { it.read(image, emptyList()) }
        Log.i(TAG, "two signs via ${reading.script}: ${reading.lines.map { "${it.text} b${it.block}" }}")
        assertEquals(listOf("EXIT", "GATE42", "TOILETS", "LIFT"), reading.lines.map { squash(it.text) })
        assertEquals(listOf(0, 0, 1, 1), reading.lines.map { it.block })
    }

    /**
     * Scripts ML Kit can't read, each sign with a Latin line under it: the chosen pack reads the native line (on
     * ML Kit's crop, or found by its own sparse scan where ML Kit Latin sees nothing), ML Kit keeps "EXIT 12".
     */
    @Test
    fun pack_scripts_read_with_their_packs_and_latin_stays_latin() {
        val cases = listOf(
            OcrLanguage.RUSSIAN to "ВЫХОД", OcrLanguage.UKRAINIAN to "ВИХІД", OcrLanguage.BULGARIAN to "ИЗХОД",
            OcrLanguage.SERBIAN to "ИЗЛАЗ", OcrLanguage.GREEK to "ΕΞΟΔΟΣ", OcrLanguage.ARABIC to "مخرج",
            OcrLanguage.HEBREW to "יציאה", OcrLanguage.THAI to "ทางออก", OcrLanguage.BENGALI to "প্রস্থান",
            OcrLanguage.TAMIL to "வெளியேறு",
        )
        val misses = ArrayList<String>()
        for ((lang, word) in cases) {
            val image = sign(300f to listOf(word, "EXIT 12"), size = 120f)
            val reading = reader(textOptions(language = lang.choice)) { it.read(image, emptyList()) }
            Log.i(TAG, "${lang.title} via ${reading.script}: ${reading.lines.map { "${it.text} (${"%.2f".format(it.conf)}, b${it.block})" }}")
            val texts = reading.lines.map { squash(it.text) }
            if (squash(word) !in texts || "EXIT12" !in texts) misses += "${lang.title}: ${reading.lines.map { it.text }}"
        }
        assertTrue(misses.joinToString("\n"), misses.isEmpty())
    }

    /** Chinese without its Play-services model (the emulator's Play services can't fetch it): the chi_sim pack. */
    @Test
    fun chinese_falls_back_to_its_pack_without_play_services_model() {
        val image = sign(300f to listOf("出口", "EXIT 12"), size = 120f)
        val reading = reader(textOptions(language = OcrLanguage.CHINESE_SIMPLIFIED.choice)) { it.read(image, emptyList()) }
        Log.i(TAG, "Chinese via ${reading.script}: ${reading.lines.map { "${it.text} (${"%.2f".format(it.conf)})" }}")
        val texts = reading.lines.map { squash(it.text) }
        assertTrue(texts.toString(), "出口" in texts && "EXIT12" in texts)
    }

    @Test
    fun auto_reads_a_latin_sign() {
        val reading = reader { it.read(sign(100f to listOf("EMERGENCY EXIT")), emptyList()) }
        assertTrue(reading.lines.map { it.text }.toString(), reading.lines.any { squash(it.text) == "EMERGENCYEXIT" })
    }

    @Test
    fun translates_per_target_and_reports_bad_targets() {
        val image = sign(100f to listOf("NOTAUSGANG", "Bitte freihalten"))
        val reading = reader { it.read(image, listOf("pl", "xx")) }
        Log.i(TAG, "translate: ${reading.lines.map { "${it.text}/${it.lang}" }} → ${reading.translations} errors ${reading.errors}")
        assertTrue(reading.lines.joinToString { "${it.text}/${it.lang}" }, reading.lines.all { it.lang == "de" })
        val pl = reading.translations["pl"]
        assertNotNull(reading.errors.toString(), pl)
        pl!!
        assertEquals(reading.lines.size, pl.size)
        assertTrue(pl.toString(), pl.any { it != null })
        assertTrue(reading.errors.toString(), "xx" in reading.errors && "xx" !in reading.translations)

        val more = reader { it.translate(listOf("Bitte freihalten", "42"), listOf(null, null), "en") }
        Log.i(TAG, "translate(): $more")
        assertEquals(2, more.size)
        assertNotNull(more[0])
        assertEquals(null, more[1]) // digits only
    }

    @Test
    fun languages_are_named_and_sorted() {
        val languages = reader { it.languages() }
        assertTrue(languages.size >= 50)
        assertTrue(languages.any { it.code == "pl" && it.name == "Polish" })
        assertEquals(languages.sortedBy { it.name }, languages)
    }

    /** Not an assertion on speed (emulators vary): logs ms per frame for the report. */
    @Test
    fun live_latency_per_frame() {
        val bus = asset("bus.jpg")
        for (side in listOf(640, 960)) {
            val (w, h) = TextLayout.fitLongSide(bus.width, bus.height, side)
            val frame = BitmapFrame(Bitmap.createScaledBitmap(bus, w, h, true))
            for (mode in listOf(LocalCatalog.LIVE_MARK, LocalCatalog.LIVE_READ)) {
                val options = textOptions(live = mode)
                val pipeline = loader.load(LocalCatalog.TEXT, options, false)
                try {
                    pipeline.run(frame, 1, options) // warm-up
                    val ms = (2..11).map { ts ->
                        val t0 = System.nanoTime()
                        pipeline.run(frame, ts.toLong(), options)
                        (System.nanoTime() - t0) / 1e6
                    }.sorted()
                    Log.i(TAG, "latency ${w}x$h $mode: median ${"%.1f".format(ms[ms.size / 2])} ms, min ${"%.1f".format(ms.first())}, max ${"%.1f".format(ms.last())}")
                } finally {
                    pipeline.close()
                }
            }
        }
    }
}
