package pl.genor.glassvision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.RectF
import android.os.LocaleList
import android.os.SystemClock
import android.util.Log
import com.google.android.gms.common.moduleinstall.ModuleInstall
import com.google.android.gms.common.moduleinstall.ModuleInstallRequest
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.nl.languageid.LanguageIdentification
import com.google.mlkit.nl.languageid.LanguageIdentifier
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.Translator
import com.google.mlkit.nl.translate.TranslatorOptions
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.googlecode.tesseract.android.TessBaseAPI
import com.googlecode.tesseract.android.TessBaseAPI.PageIteratorLevel
import java.util.EnumMap
import java.util.concurrent.TimeUnit

/*
 * On-device text: ML Kit Text Recognition v2 finds and reads the lines (scene text, unlike Tesseract's page
 * layout analysis, which reads almost nothing on a camera frame); Tesseract only reads ML Kit's line crops for
 * scripts ML Kit has no recognizer for. The pure parts (reading order, crops, plans, languages) are in
 * TextLayout.kt. API: https://developers.google.com/ml-kit/vision/text-recognition/v2/android,
 * https://developers.google.com/ml-kit/language/translation/android, https://github.com/adaptech-cz/Tesseract4Android
 */

/** The phone's languages (system settings order) as bare codes ("pl", "he"…), for [OcrPlan.forChoice]. */
internal fun phoneLanguages(): List<String> {
    val locales = LocaleList.getDefault()
    return (0 until locales.size()).map { locales[it].toLanguageTag().substringBefore('-') }.distinct()
}

/** The plan for the text pipeline's [options] on this phone (installed packs, phone languages). */
internal fun planFor(options: Map<String, OptionValue>, models: ModelManager): OcrPlan {
    val choice = (options[LocalCatalog.OPT_OCR_LANGUAGE] as? OptionValue.Str)?.value ?: OcrLanguage.AUTO_CHOICE
    val installed = OcrLanguage.entries.filter { models.isReady(it.spec) }.sortedByDescending { models.file(it.spec).lastModified() }
    return OcrPlan.forChoice(choice, installed, phoneLanguages())
}

/**
 * ML Kit text recognizers by script, created on first use. Latin is bundled; the others come from Google Play
 * services: [available] asks Play services to download a missing one and says no until it is there.
 */
internal class MlKitReaders(context: Context) : AutoCloseable {
    private val context = context.applicationContext
    private val clients = EnumMap<TextScript, TextRecognizer>(TextScript::class.java)
    private val ready = EnumMap<TextScript, Boolean>(TextScript::class.java)
    private val recheckAt = EnumMap<TextScript, Long>(TextScript::class.java)

    private fun client(script: TextScript): TextRecognizer = synchronized(clients) {
        clients.getOrPut(script) {
            TextRecognition.getClient(
                when (script) {
                    TextScript.LATIN, TextScript.OTHER -> TextRecognizerOptions.DEFAULT_OPTIONS
                    TextScript.CHINESE -> ChineseTextRecognizerOptions.Builder().build()
                    TextScript.JAPANESE -> JapaneseTextRecognizerOptions.Builder().build()
                    TextScript.KOREAN -> KoreanTextRecognizerOptions.Builder().build()
                    TextScript.DEVANAGARI -> DevanagariTextRecognizerOptions.Builder().build()
                },
            )
        }
    }

    /** Blocking (Play services call, at most every [RECHECK_MS] per script while missing). */
    fun available(script: TextScript): Boolean {
        if (script == TextScript.LATIN) return true
        if (script == TextScript.OTHER) return false
        val now = SystemClock.elapsedRealtime()
        synchronized(ready) {
            if (ready[script] == true) return true
            if (now < (recheckAt[script] ?: 0L)) return false
            recheckAt[script] = now + RECHECK_MS
        }
        val api = client(script)
        val modules = ModuleInstall.getClient(context)
        val ok = try {
            Tasks.await(modules.areModulesAvailable(api), TIMEOUT_S, TimeUnit.SECONDS).areModulesAvailable()
        } catch (e: Exception) {
            Log.w(TAG, "${script.title}: Play services module check failed", e)
            false
        }
        if (ok) {
            synchronized(ready) { ready[script] = true }
        } else {
            try {
                modules.installModules(ModuleInstallRequest.Builder().addApi(api).build())
            } catch (e: Exception) {
                Log.w(TAG, "${script.title}: Play services module install request failed", e)
            }
        }
        return ok
    }

    /** All lines of [bitmap] (upright) in pixels; [OcrLine.block] = ML Kit's text block. */
    fun read(script: TextScript, bitmap: Bitmap): List<OcrLine> {
        val text = Tasks.await(client(script).process(InputImage.fromBitmap(bitmap, 0)), TIMEOUT_S, TimeUnit.SECONDS)
        return lines(text, script)
    }

    override fun close() {
        synchronized(clients) {
            clients.values.forEach(TextRecognizer::close)
            clients.clear()
        }
    }

    companion object {
        private const val TAG = "MlKitReaders"
        const val TIMEOUT_S = 30L
        const val RECHECK_MS = 30_000L

        /** ML Kit's lines with their corner points (clockwise from the text's top-left) and language tag. */
        fun lines(text: Text, script: TextScript): List<OcrLine> = text.textBlocks.flatMapIndexed { b, block ->
            block.lines.mapNotNull { line ->
                val r = line.boundingBox ?: return@mapNotNull null
                val t = line.text.trim()
                if (t.isEmpty() || r.width() <= 0 || r.height() <= 0) return@mapNotNull null
                val quad = line.cornerPoints?.takeIf { it.size == 4 }?.let { p -> Quad(p.map { Pt(it.x.toFloat(), it.y.toFloat()) }) }
                OcrLine(
                    t, line.confidence.coerceIn(0f, 1f), TextBox(r.left.toFloat(), r.top.toFloat(), r.right.toFloat(), r.bottom.toFloat()), b,
                    quad = quad, lang = line.recognizedLanguage, script = script.name.lowercase(),
                )
            }
        }
    }
}

/**
 * Tesseract on single line crops (grayscale, scaled to ~48 px text), and [sparse] whole-image scans for scripts
 * ML Kit doesn't find. Not thread-safe: one owner thread at a time.
 */
internal class TesseractCrops(private val models: ModelManager) : AutoCloseable {
    private val apis = LinkedHashMap<OcrLanguage, TessBaseAPI>(4, 0.75f, true)

    /** Downloads [lang]'s pack if needed (verified, see ModelManager) and loads it; throws with a readable message. */
    fun prepare(lang: OcrLanguage): TessBaseAPI {
        apis[lang]?.let { return it }
        val model = models.ensure(lang.spec)
        model.setLastModified(System.currentTimeMillis()) // "recently used" for auto mode's pack order (planFor)
        val api = TessBaseAPI()
        val dataRoot = requireNotNull(model.parentFile?.parentFile) { "Invalid OCR model path" }
        if (!api.init(dataRoot.absolutePath, lang.code, TessBaseAPI.OEM_LSTM_ONLY)) {
            api.recycle()
            throw ModelUnavailableException("OCR model could not be loaded: ${lang.title}")
        }
        apis[lang] = api
        while (apis.size > MAX_LOADED) apis.entries.iterator().run { next().value.also { remove() } }.recycle()
        return api
    }

    /** [lang]'s reading of [line] on [image], or null if the crop is degenerate. */
    fun read(lang: OcrLanguage, image: Bitmap, line: TextBox): Reading? {
        val api = prepare(lang)
        val crop = TextLayout.crop(line, image.width, image.height) ?: return null
        val (w, h) = TextLayout.scaled(crop.width, crop.height, TextLayout.cropScale(line.height))
        val gray = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Canvas(gray).drawBitmap(
            image,
            android.graphics.Rect(crop.left, crop.top, crop.right, crop.bottom),
            android.graphics.Rect(0, 0, w, h),
            GRAY,
        )
        return try {
            api.setPageSegMode(TessBaseAPI.PageSegMode.PSM_SINGLE_LINE)
            api.setImage(gray)
            val text = api.utF8Text?.trim().orEmpty()
            Reading(text.replace(WHITESPACE, " "), (api.meanConfidence() / 100f).coerceIn(0f, 1f))
        } finally {
            api.clear()
            gray.recycle()
        }
    }

    /**
     * [lang]'s lines on the whole [image] with page segmentation [psm] (PSM_SPARSE_TEXT: text anywhere, no page
     * layout), read at ≤ [SPARSE_LONG_SIDE] px, in pixels of [image]; blocks numbered from [firstBlock].
     */
    fun scan(lang: OcrLanguage, image: Bitmap, firstBlock: Int, psm: Int = TessBaseAPI.PageSegMode.PSM_SPARSE_TEXT): List<OcrLine> {
        val api = prepare(lang)
        val (w, h) = TextLayout.fitLongSide(image.width, image.height, SPARSE_LONG_SIDE)
        val scaled = if (w == image.width && h == image.height) image else Bitmap.createScaledBitmap(image, w, h, true)
        val sx = image.width.toFloat() / w
        val sy = image.height.toFloat() / h
        try {
            api.setPageSegMode(psm)
            api.setImage(scaled)
            // The iterator is null until recognition ran: getUTF8Text() runs it.
            if (api.utF8Text.isNullOrBlank()) return emptyList()
            val iterator = api.resultIterator ?: return emptyList()
            val lines = ArrayList<OcrLine>()
            try {
                do {
                    val text = iterator.getUTF8Text(PageIteratorLevel.RIL_TEXTLINE)?.trim()?.replace(WHITESPACE, " ").orEmpty()
                    val r = iterator.getBoundingRect(PageIteratorLevel.RIL_TEXTLINE)
                    if (text.isEmpty() || r.width() <= 0 || r.height() <= 0) continue
                    val conf = (iterator.confidence(PageIteratorLevel.RIL_TEXTLINE) / 100f).coerceIn(0f, 1f)
                    val box = TextBox(r.left * sx, r.top * sy, r.right * sx, r.bottom * sy)
                    lines += OcrLine(text, conf, box, firstBlock + lines.size, script = lang.code)
                } while (iterator.next(PageIteratorLevel.RIL_TEXTLINE))
            } finally {
                iterator.delete()
            }
            return lines
        } finally {
            api.clear()
            if (scaled !== image) scaled.recycle()
        }
    }

    override fun close() {
        apis.values.forEach(TessBaseAPI::recycle)
        apis.clear()
    }

    private companion object {
        const val MAX_LOADED = 3
        const val SPARSE_LONG_SIDE = 1600
        val WHITESPACE = Regex("\\s+")
        val GRAY = Paint(Paint.FILTER_BITMAP_FLAG).apply { colorFilter = ColorMatrixColorFilter(ColorMatrix().apply { setSaturation(0f) }) }
    }
}

/**
 * Reads lines following an [OcrPlan]: the plan's ML Kit recognizer (ML Kit Latin while it is unavailable), then the
 * plan's Tesseract packs on line crops where they apply. One owner thread at a time.
 */
internal class TextEngine(private val mlKit: MlKitReaders, private val tesseract: TesseractCrops) : AutoCloseable {
    /** What read the last image, for display ("ML Kit Latin + Tesseract rus"). */
    @Volatile var lastEngine: String? = null
        private set

    /**
     * Lines in pixels of [bitmap], in engine order (blocks as found). [live]: a camera frame — Tesseract crops are
     * capped per frame, and only an explicitly chosen pack scans the whole frame (sparse text only). [boxesOnly]:
     * the text isn't used (live "Mark text areas"), so lines ML Kit found aren't re-read.
     */
    fun read(bitmap: Bitmap, plan: OcrPlan, live: Boolean = false, boxesOnly: Boolean = false): List<OcrLine> {
        val available = mlKit.available(plan.script)
        val script = if (available) plan.script else TextScript.LATIN
        val lines = try {
            mlKit.read(script, bitmap)
        } catch (e: Exception) {
            if (script == TextScript.LATIN) throw e
            // A Play-services recognizer that still fails (module half-installed): Latin + the fallback packs.
            Log.w(TAG, "${script.title} recognizer failed, using Latin", e)
            return readWith(bitmap, plan, TextScript.LATIN, mlKit.read(TextScript.LATIN, bitmap), fallback = true, live, boxesOnly)
        }
        return readWith(bitmap, plan, script, lines, fallback = !available, live, boxesOnly)
    }

    private fun readWith(
        bitmap: Bitmap,
        plan: OcrPlan,
        script: TextScript,
        found: List<OcrLine>,
        fallback: Boolean,
        live: Boolean,
        boxesOnly: Boolean,
    ): List<OcrLine> {
        val packs = (if (fallback) plan.fallback + plan.packs else plan.packs).distinct()
        val used = LinkedHashSet<OcrLanguage>()
        fun <T> attempt(lang: OcrLanguage, block: () -> T): T? = try {
            block().also { used += lang }
        } catch (e: Exception) {
            if (!plan.auto) throw e // the user's chosen pack: its download error is the answer
            Log.w(TAG, "auto: skipping ${lang.title}", e)
            null
        }

        var budget = if (boxesOnly) 0 else if (live) LIVE_MAX_CROPS else Int.MAX_VALUE
        val lines = found.map { line ->
            if (packs.isEmpty() || (!fallback && line.conf >= plan.retryBelow) || budget < packs.size) return@map line
            budget -= packs.size
            val readings = packs.mapNotNull { lang -> attempt(lang) { tesseract.read(lang, bitmap, line.box) }?.let { lang to it } }
            val best = TextLanguages.choose(Reading(line.text, line.conf), readings)
            val pack = readings.firstOrNull { it.second === best }?.first
            line.copy(text = best.text, conf = best.conf, script = pack?.code ?: line.script)
        }.toMutableList()

        // Scripts ML Kit Latin doesn't box at all: the pack looks for its own lines where ML Kit found none. Sparse
        // text first; on a still, page segmentation if that found nothing (it misses short words, e.g. one Arabic word).
        if (!(live && plan.auto)) {
            val modes = listOf(TessBaseAPI.PageSegMode.PSM_SPARSE_TEXT) + if (live) emptyList() else listOf(TessBaseAPI.PageSegMode.PSM_AUTO)
            for ((i, lang) in packs.filter { !it.latinFindsLines }.withIndex()) {
                for ((m, psm) in modes.withIndex()) {
                    val firstBlock = SPARSE_BLOCKS * (1 + i * modes.size + m)
                    val extra = attempt(lang) { tesseract.scan(lang, bitmap, firstBlock, psm) } ?: break
                    val new = extra.filter { line ->
                        line.conf >= TextLanguages.MIN_PACK_CONF && TextLanguages.inScript(line.text, lang) &&
                            lines.none { TextLayout.covered(line.box, it.box) > SPARSE_OVERLAP }
                    }
                    lines += new
                    if (new.isNotEmpty()) break
                }
            }
        }
        lastEngine = "ML Kit ${script.title}" + if (used.isEmpty()) "" else " + Tesseract ${used.joinToString(",") { it.code }}"
        return lines
    }

    override fun close() {
        mlKit.close()
        tesseract.close()
    }

    private companion object {
        const val TAG = "TextEngine"
        /** Tesseract crop reads (line × pack) per live frame, ~10–60 ms each; a still reads every line. */
        const val LIVE_MAX_CROPS = 8
        /** Block numbers of a pack's sparse lines start at multiples of this, apart from ML Kit's. */
        const val SPARSE_BLOCKS = 100_000
        /** A sparse line mostly inside a line ML Kit found is the same line (already read). */
        const val SPARSE_OVERLAP = 0.3f
    }
}

/** Language identification and ML Kit offline translation with caches; models download on first use. Thread-safe. */
internal class Translations : AutoCloseable {
    private val identifier: LanguageIdentifier = LanguageIdentification.getClient()
    private val translators = LinkedHashMap<Pair<String, String>, Translator>(8, 0.75f, true)
    private val translated = LinkedHashMap<String, String>(256, 0.75f, true)
    private val identified = LinkedHashMap<String, String>(256, 0.75f, true)

    /** Source code of [text] or null (undetermined, romanized, failed). */
    fun identify(text: String): String? {
        synchronized(identified) { identified[text]?.let { return TextLanguages.sourceCode(it) } }
        val tag = try {
            Tasks.await(identifier.identifyLanguage(text), TIMEOUT_S, TimeUnit.SECONDS)
        } catch (_: Exception) {
            return null
        }
        synchronized(identified) {
            identified[text] = tag
            trim(identified)
        }
        return TextLanguages.sourceCode(tag)
    }

    /** [text] from [source] into [target]; throws if a model can't be downloaded or the pair isn't supported. */
    fun translate(text: String, source: String, target: String): String? {
        val from = TranslateLanguage.fromLanguageTag(source) ?: return null // ML Kit can't translate from it
        val to = TranslateLanguage.fromLanguageTag(target) ?: throw IllegalArgumentException("Unsupported language: $target")
        val key = "$from\u0000$to\u0000$text"
        synchronized(translated) { translated[key]?.let { return it } }
        val translator = translator(from, to)
        // Lazy on purpose: a target's models download the first time a line needs them.
        Tasks.await(translator.downloadModelIfNeeded(), DOWNLOAD_TIMEOUT_S, TimeUnit.SECONDS)
        val result = Tasks.await(translator.translate(text), TIMEOUT_S, TimeUnit.SECONDS).trim()
        if (result.isEmpty()) return null
        synchronized(translated) {
            translated[key] = result
            trim(translated)
        }
        return result
    }

    private fun translator(source: String, target: String): Translator = synchronized(translators) {
        translators.getOrPut(source to target) {
            Translation.getClient(TranslatorOptions.Builder().setSourceLanguage(source).setTargetLanguage(target).build())
        }.also {
            while (translators.size > MAX_TRANSLATORS) translators.entries.iterator().run { next().value.also { remove() } }.close()
        }
    }

    private fun trim(cache: LinkedHashMap<String, String>) {
        while (cache.size > MAX_CACHE) cache.entries.iterator().run { next(); remove() }
    }

    override fun close() {
        identifier.close()
        synchronized(translators) {
            translators.values.forEach(Translator::close)
            translators.clear()
        }
    }

    companion object {
        const val TIMEOUT_S = 10L
        /** A language pair is ~30 MB on first use. */
        const val DOWNLOAD_TIMEOUT_S = 120L
        const val MAX_CACHE = 512
        const val MAX_TRANSLATORS = 8

        /** ML Kit's translation languages. */
        fun languages(): List<TranslationLanguage> = TextLanguages.targets(TranslateLanguage.getAllLanguages())
    }
}

/**
 * The live text pipeline (slow lane). Option `live`: "Mark text areas" → one box per block ([TextBlocks]: a sign,
 * a paragraph) labelled "text"; "Read live" → per line its text, or its translation (`translate_to`, original as
 * detail) like before.
 */
class TextPipeline internal constructor(
    private val engine: TextEngine,
    private val plan: OcrPlan,
    private val translations: Translations,
    override val note: String? = null,
) : LocalPipeline {
    override fun run(frame: LocalFrame, timestampMs: Long, options: Map<String, OptionValue>): List<PixelDetection> {
        val f = frame as? BitmapFrame ?: throw IllegalArgumentException("on-device pipelines need a BitmapFrame")
        val readLive = (options[LocalCatalog.OPT_LIVE] as? OptionValue.Str)?.value == LocalCatalog.LIVE_READ
        val grouping = TextBlocks.group(engine.read(f.bitmap, plan, live = true, boxesOnly = !readLive))
        if (!readLive) {
            return grouping.blocks.mapNotNull { b ->
                val conf = b.lines.maxOf { grouping.lines[it].conf }
                pixels(b.quad.bounds(), conf, f, LocalCatalog.TEXT_LABEL, null)
            }
        }
        val lines = grouping.lines
        val target = LocalCatalog.translationTag((options[LocalCatalog.OPT_TRANSLATE_TO] as? OptionValue.Str)?.value ?: LocalCatalog.TRANSLATE_OFF)
        val sources = if (target == null) null else TextLanguages.assign(lines, translations::identify)
        return lines.mapIndexedNotNull { i, line ->
            val source = sources?.get(i)
            val translation = if (target != null && TextLanguages.translatable(line.text, source, target)) {
                try {
                    translations.translate(line.text, source!!, target)?.takeIf { it != line.text }
                } catch (_: Exception) {
                    null // keep the raw OCR line: a failed model download must not take the pipeline down
                }
            } else {
                null
            }
            pixels(line.box, line.conf, f, translation ?: line.text, line.text.takeIf { translation != null })
        }
    }

    private fun pixels(box: TextBox, conf: Float, f: LocalFrame, label: String, detail: String?): PixelDetection? {
        val l = box.left.coerceIn(0f, f.width.toFloat())
        val t = box.top.coerceIn(0f, f.height.toFloat())
        val r = box.right.coerceIn(0f, f.width.toFloat())
        val b = box.bottom.coerceIn(0f, f.height.toFloat())
        if (r <= l || b <= t) return null
        return PixelDetection(l, t, r, b, conf, label, detail)
    }

    override fun close() {
        engine.close()
        translations.close()
    }
}

/**
 * The Read button on this phone ([TextReader]): ML Kit v2 on the full still (≤ 2560 px long side), Tesseract on
 * line crops per the text pipeline's OCR language ([options], read on each call), lines grouped into blocks in
 * reading order ([TextBlocks]), per-line language ID, ML Kit translation per block and target (prose as one text,
 * lists line by line). One read at a time; [close] waits for a running one to finish before releasing the models.
 */
class LocalTextReader(
    context: Context,
    private val models: ModelManager,
    private val options: () -> Map<String, OptionValue>,
) : TextReader, AutoCloseable {
    private val engine = TextEngine(MlKitReaders(context), TesseractCrops(models))
    private val translations = Translations()
    private val state = Any()
    private val work = Any()
    private var busy = 0
    private var closed = false

    override fun read(still: Bitmap, translateTo: List<String>): TextReading = session {
        val (w, h) = TextLayout.fitLongSide(still.width, still.height)
        val image = if (w == still.width && h == still.height) still else Bitmap.createScaledBitmap(still, w, h, true)
        try {
            val grouping = TextBlocks.group(engine.read(image, planFor(options(), models)))
            val sources = TextLanguages.assign(grouping.lines, translations::identify)
            val read = LocalReading.of(grouping, sources, w.toFloat(), h.toFloat())
            val done = LinkedHashMap<String, List<String?>>()
            val errors = LinkedHashMap<String, String>()
            for (target in translateTo.distinct()) {
                try {
                    done[target] = translateBlockTexts(read.blocks, target)
                } catch (e: Exception) {
                    errors[target] = LocalBackend.readable(e)
                }
            }
            val blocks = read.blocks.mapIndexed { i, b -> b.copy(translations = done.mapValues { it.value[i] }) }
            TextReading(still.width, still.height, read.lines, blocks, errors, engine.lastEngine, grouping.angle)
        } finally {
            if (image !== still) image.recycle()
        }
    }

    override fun translate(lines: List<String>, from: List<String?>, to: String): List<String?> = session {
        val detected = TextLanguages.assign(lines.map { OcrLine(it, 1f, TextBox(0f, 0f, 1f, 1f), 0) }, translations::identify)
        translateAll(lines, lines.indices.map { from.getOrNull(it) ?: detected[it] }, to)
    }

    override fun translateBlocks(blocks: List<ReadBlock>, to: String): List<String?> = session { translateBlockTexts(blocks, to) }

    override fun languages(): List<TranslationLanguage> = Translations.languages()

    /** Null where a line isn't translated; throws if every line that should have been failed (the target is out). */
    private fun translateAll(lines: List<String>, sources: List<String?>, target: String): List<String?> {
        if (TranslateLanguage.fromLanguageTag(target) == null) throw IllegalArgumentException("Unsupported language: $target")
        val tally = Tally()
        val out = lines.mapIndexed { i, text -> tally.translate(text, sources[i], target) }
        tally.check()
        return out
    }

    /**
     * Per block: prose joined (de-hyphenated) and translated as one text, a list line by line ([BlockKind]); source =
     * the block's language, else identified on the block's text. Throws if every attempt failed.
     */
    private fun translateBlockTexts(blocks: List<ReadBlock>, target: String): List<String?> {
        if (TranslateLanguage.fromLanguageTag(target) == null) throw IllegalArgumentException("Unsupported language: $target")
        val tally = Tally()
        val out = blocks.map { b ->
            val lines = b.text.split('\n')
            val source = b.lang ?: b.text.takeIf { TextLanguages.letters(it) >= TextLanguages.MIN_ID_LETTERS }?.let(translations::identify)
            if (b.kind == BlockKind.PROSE) {
                tally.translate(TextBlocks.joinProse(lines), source, target)
            } else {
                val each = lines.map { tally.translate(it, source, target) }
                if (each.all { it == null }) null else lines.indices.joinToString("\n") { each[it] ?: lines[it] }
            }
        }
        tally.check()
        return out
    }

    /** Counts translation attempts so a target that failed for every text throws its error instead of looking unchanged. */
    private inner class Tally {
        private var attempted = 0
        private var failed = 0
        private var failure: Exception? = null

        fun translate(text: String, source: String?, target: String): String? {
            if (!TextLanguages.translatable(text, source, target)) return null
            attempted++
            return try {
                translations.translate(text, source!!, target)?.takeIf { it != text }
            } catch (e: Exception) {
                failed++
                failure = e
                null
            }
        }

        fun check() {
            failure?.let { if (failed == attempted) throw it }
        }
    }

    private fun <T> session(block: () -> T): T {
        synchronized(state) {
            check(!closed) { "Text reader is closed" }
            busy++
        }
        try {
            return synchronized(work) { block() }
        } finally {
            val release = synchronized(state) { --busy == 0 && closed }
            if (release) release()
        }
    }

    override fun close() {
        val release = synchronized(state) {
            if (closed) return
            closed = true
            busy == 0
        }
        if (release) release()
    }

    private fun release() {
        engine.close()
        translations.close()
    }
}
