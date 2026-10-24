package pl.genor.glassvision

import android.graphics.Bitmap
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit

/** The server's text pipeline settings a read uses (option values exactly as in `hello`; null = server default). */
data class ReadOptions(
    /** `languages` choice (script group / auto-detect). */
    val languages: String? = null,
    /** `translation_model` choice. */
    val model: String? = null,
    /** `translate_from` choice: a language *name* ("German") or "Auto-detect"; mapped to a code via [TextReader.languages]. */
    val translateFrom: String? = null,
)

/** A read/translate request the server refused or that never got an answer; [message] is shown as is. */
class TextReadException(message: String, cause: Throwable? = null) : IOException(message, cause)

/**
 * [TextReader] over the server's HTTP API (docs/PROTOCOL.md "Read text"): `POST /text/read` with the still as JPEG,
 * `POST /text/translate`, `GET /text/languages` (cached). Same token as everything else ([ServerEndpoint.authorization]).
 * [options] is read per call, so a changed text-pipeline setting applies to the next read. [features] = `hello.features`:
 * with `text_blocks` (v3.3) reads ask for `format=blocks` and "Add language" translates whole blocks. Blocking; worker
 * threads only.
 */
class ServerTextReader(
    private val endpoint: ServerEndpoint,
    private val options: () -> ReadOptions = { ReadOptions() },
    private val client: OkHttpClient = defaultClient,
    private val features: Set<String> = emptySet(),
) : TextReader, JpegTextReader {
    private val blocks: Boolean get() = Protocol.FEATURE_TEXT_BLOCKS in features
    @Volatile private var languagesCache: List<TranslationLanguage>? = null

    override fun read(still: Bitmap, translateTo: List<String>): TextReading = readJpeg(encode(still), translateTo)

    /** [read] for a JPEG as is: upright, or with an EXIF orientation (the server applies it; [Still.uploadJpeg]). */
    override fun readJpeg(jpeg: ByteArray, translateTo: List<String>): TextReading {
        val opts = options()
        val url = url("/text/read").newBuilder().apply {
            opts.languages?.takeIf { it.isNotBlank() }?.let { addQueryParameter("languages", it) }
            if (translateTo.isNotEmpty()) addQueryParameter("translate_to", translateTo.joinToString(","))
            fromCode(opts)?.let { addQueryParameter("translate_from", it) }
            opts.model?.takeIf { it.isNotBlank() }?.let { addQueryParameter("model", it) }
            if (blocks) addQueryParameter("format", "blocks")
        }.build()
        return TextReadJson.reading(call(url, jpeg.toRequestBody(JPEG)))
    }

    override fun translate(lines: List<String>, from: List<String?>, to: String): List<String?> {
        if (lines.isEmpty()) return emptyList()
        val body = JSONObject()
            .put("lines", JSONArray(lines))
            .put("from", JSONArray().also { arr -> lines.indices.forEach { arr.put(from.getOrNull(it) ?: JSONObject.NULL) } })
            .put("to", to)
            .put("model", options().model?.takeIf { it.isNotBlank() } ?: JSONObject.NULL)
        val json = call(url("/text/translate"), body.toString().toRequestBody(JSON))
        return TextReadJson.translations(json, lines.size)
    }

    /** v3.3: `{"blocks": [{text, lang, kind}], "to"}`, prose blocks translated as one text; older servers: per line. */
    override fun translateBlocks(blocks: List<ReadBlock>, to: String): List<String?> {
        if (!this.blocks) return super.translateBlocks(blocks, to)
        if (blocks.isEmpty()) return emptyList()
        val body = JSONObject()
            .put("blocks", JSONArray(blocks.map { b -> JSONObject().put("text", b.text).put("lang", b.lang ?: JSONObject.NULL).put("kind", b.kind.wire) }))
            .put("to", to)
            .put("model", options().model?.takeIf { it.isNotBlank() } ?: JSONObject.NULL)
        return TextReadJson.translations(call(url("/text/translate"), body.toString().toRequestBody(JSON)), blocks.size)
    }

    override fun languages(): List<TranslationLanguage> {
        languagesCache?.let { return it }
        val list = TextReadJson.languages(call(url("/text/languages"), null))
        languagesCache = list
        return list
    }

    /** `translate_from` as a code: the option is a language name, the endpoint takes a code; unknown → detect. */
    private fun fromCode(opts: ReadOptions): String? {
        val name = opts.translateFrom?.takeIf { it.isNotBlank() } ?: return null
        val known = runCatching { languages() }.getOrDefault(emptyList())
        return known.firstOrNull { it.name.equals(name, ignoreCase = true) || it.code == name }?.code
    }

    private fun url(path: String): HttpUrl {
        val base = endpoint.httpBase ?: throw TextReadException("No server address")
        return (base + path).toHttpUrlOrNull() ?: throw TextReadException("Bad server address: $base")
    }

    /** GET when [body] is null, else POST; the parsed JSON object, or [TextReadException] with the server's reason. */
    private fun call(url: HttpUrl, body: RequestBody?): JSONObject {
        val request = Request.Builder().url(url).apply {
            endpoint.authorization()?.let { header("Authorization", it) }
            if (body != null) post(body)
        }.build()
        try {
            client.newCall(request).execute().use { response ->
                val text = response.body?.string().orEmpty()
                if (!response.isSuccessful) throw TextReadException(TextReadJson.errorMessage(response.code, text))
                return try {
                    JSONObject(text)
                } catch (e: JSONException) {
                    throw TextReadException("Unexpected answer from the server", e)
                }
            }
        } catch (e: TextReadException) {
            throw e
        } catch (e: IOException) {
            throw TextReadException("Server not reachable: ${e.message ?: e.javaClass.simpleName}", e)
        }
    }

    private fun encode(still: Bitmap): ByteArray {
        // The server reads at up to 2560 px anyway: don't upload more.
        val scaled = FrameEncoder.uprightBitmap(still, 0, MAX_UPLOAD_LONG_SIDE)
        try {
            val out = ByteArrayOutputStream(1 shl 20)
            if (!scaled.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)) throw TextReadException("JPEG encoding failed")
            return out.toByteArray()
        } finally {
            if (scaled !== still) scaled.recycle()
        }
    }

    companion object {
        const val MAX_UPLOAD_LONG_SIDE = 2560
        const val JPEG_QUALITY = 90
        private val JPEG = "image/jpeg".toMediaType()
        private val JSON = "application/json".toMediaType()

        /** OCR of a large still (+ a model load on first use) takes longer than a health check. */
        val defaultClient: OkHttpClient by lazy {
            ServerHttp.client.newBuilder()
                .writeTimeout(30, TimeUnit.SECONDS)
                .readTimeout(90, TimeUnit.SECONDS)
                .build()
        }
    }
}

/**
 * JSON of the Read text endpoints → [TextReading] & co. Lenient like [Protocol]: bad entries are skipped. A v3.3
 * answer (`blocks`) is taken as is; a v3.2 one gets its blocks from the lines' `block` ids ([ReadBlocks.fromLines]).
 */
object TextReadJson {
    fun reading(o: JSONObject): TextReading {
        // Line indices (block members, v3.2 translations) refer to the server's lines: map them past skipped ones.
        val raw = o.optJSONArray("lines")
        val kept = ArrayList<Int>()
        val index = HashMap<Int, Int>()
        val lines = ArrayList<ReadLine>()
        for (i in 0 until (raw?.length() ?: 0)) {
            val line = raw?.optJSONObject(i)?.let { line(it, i) } ?: continue
            index[i] = lines.size
            lines += line
            kept += i
        }
        val errors = LinkedHashMap<String, String>()
        o.optJSONObject("errors")?.let { e -> for (code in e.keys()) errors[code] = e.optString(code).ifEmpty { "failed" } }
        val parsed = o.optJSONArray("blocks").objects().mapIndexedNotNull { i, b -> block(b, i, index, lines) }
        val blocks = parsed.ifEmpty {
            val translations = LinkedHashMap<String, List<String?>>()
            o.optJSONObject("translations")?.let { t ->
                for (code in t.keys()) t.optJSONArray(code)?.let { arr -> translations[code] = kept.map { entry(arr, it) } }
            }
            if (lines.isEmpty()) emptyList() else ReadBlocks.fromLines(lines, translations)
        }
        return TextReading(
            width = o.optInt("w", 0),
            height = o.optInt("h", 0),
            lines = lines,
            blocks = blocks,
            errors = errors,
            script = o.optStringOrNull("script"),
            angle = o.optDouble("angle", 0.0).toFloat().takeIf { it.isFinite() } ?: 0f,
        )
    }

    /** `/text/translate` answer, aligned to [count] input lines (missing entries = untranslated). */
    fun translations(o: JSONObject, count: Int): List<String?> {
        val arr = o.optJSONArray("translations") ?: throw TextReadException("Unexpected answer from the server")
        return aligned(arr, count)
    }

    fun languages(o: JSONObject): List<TranslationLanguage> =
        o.optJSONArray("languages").objects().mapNotNull { l ->
            val code = l.optString("code").trim().ifEmpty { return@mapNotNull null }
            TranslationLanguage(code, l.optString("name").trim().ifEmpty { code })
        }.distinctBy { it.code }.sortedBy { it.name.lowercase() }

    /** What to tell the user for a non-2xx answer: FastAPI's `detail` when it is a string, else the status. */
    fun errorMessage(status: Int, body: String): String {
        val detail = try {
            JSONObject(body).opt("detail") as? String
        } catch (_: JSONException) {
            null
        }?.trim()?.takeIf { it.isNotEmpty() }
        return when (status) {
            401, 403 -> "Access token rejected (HTTP $status)"
            404 -> "This server cannot read text (HTTP 404)"
            else -> if (detail != null) "$detail (HTTP $status)" else "Server error (HTTP $status)"
        }
    }

    private fun line(l: JSONObject, index: Int): ReadLine? {
        val text = l.optString("text").trim().ifEmpty { return null }
        return ReadLine(
            text = text,
            conf = l.optDouble("conf", 0.0).toFloat(),
            box = box(l.optJSONArray("box")) ?: TextBox(0f, 0f, 0f, 0f),
            block = if (l.has("block") && !l.isNull("block")) l.optInt("block", index) else index,
            lang = l.optStringOrNull("lang"),
            quad = quad(l.optJSONArray("quad")),
            script = l.optStringOrNull("script"),
            suspect = l.optBoolean("suspect", false),
        )
    }

    /** A v3.3 block; null if none of its lines survived. [index] = server line index → ours. */
    private fun block(b: JSONObject, position: Int, index: Map<Int, Int>, lines: List<ReadLine>): ReadBlock? {
        val members = b.optJSONArray("lines").let { arr -> (0 until (arr?.length() ?: 0)).mapNotNull { arr?.opt(it) as? Number } }
            .mapNotNull { index[it.toInt()] }.distinct()
        if (members.isEmpty()) return null
        val texts = members.map { lines[it].text }
        val box = box(b.optJSONArray("box")) ?: members.map { lines[it].box }.reduce(TextBox::union)
        val translations = LinkedHashMap<String, String?>()
        b.optJSONObject("translations")?.let { t ->
            for (code in t.keys()) translations[code] = if (t.isNull(code)) null else (t.opt(code) as? String)?.takeIf { it.isNotBlank() }
        }
        return ReadBlock(
            id = b.optInt("id", position),
            lines = members,
            box = box,
            quad = quad(b.optJSONArray("quad")),
            text = b.optStringOrNull("text")?.trim()?.takeIf { it.isNotEmpty() } ?: texts.joinToString("\n"),
            kind = BlockKind.of(b.optStringOrNull("kind")) ?: TextBlocks.kind(texts, members.map { lines[it].box.width }, box.width),
            lang = b.optStringOrNull("lang"),
            suspect = b.optBoolean("suspect", false),
            translations = translations,
        )
    }

    private fun box(b: JSONArray?): TextBox? {
        if (b == null || b.length() < 4) return null
        return TextBox(b.optDouble(0, 0.0).toFloat(), b.optDouble(1, 0.0).toFloat(), b.optDouble(2, 0.0).toFloat(), b.optDouble(3, 0.0).toFloat())
    }

    /** `[[x, y] × 4]`, or null if it isn't exactly that. */
    private fun quad(q: JSONArray?): Quad? {
        if (q == null || q.length() != 4) return null
        val points = (0 until 4).map { i ->
            val p = q.optJSONArray(i) ?: return null
            if (p.length() < 2) return null
            val x = p.optDouble(0, Double.NaN)
            val y = p.optDouble(1, Double.NaN)
            if (x.isNaN() || y.isNaN()) return null
            Pt(x.toFloat(), y.toFloat())
        }
        return Quad(points)
    }

    private fun aligned(arr: JSONArray, count: Int): List<String?> = List(count) { entry(arr, it) }

    private fun entry(arr: JSONArray, i: Int): String? =
        if (i >= arr.length() || arr.isNull(i)) null else (arr.opt(i) as? String)?.takeIf { it.isNotBlank() }

    private fun JSONArray?.objects(): List<JSONObject> =
        if (this == null) emptyList() else (0 until length()).mapNotNull { optJSONObject(it) }

    private fun JSONObject.optStringOrNull(key: String): String? =
        if (has(key) && !isNull(key)) optString(key).takeIf { it.isNotEmpty() } else null
}
