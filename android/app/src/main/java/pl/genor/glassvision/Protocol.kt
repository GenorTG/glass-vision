package pl.genor.glassvision

import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** A keypoint, normalized like [NormBox]; [v] = visibility/confidence 0..1. */
data class KeyPoint(val x: Float, val y: Float, val v: Float)

/**
 * One detection or smoothed track of one [pipeline]; [box] is `[x1, y1, x2, y2]` normalized 0..1 against the
 * upright JPEG that was sent. Tracks carry [trackId] (`"<pipeline>:<n>"`) and [seenMsAgo] (0 = detected in
 * this frame, >0 = held/predicted).
 */
data class Detection(
    val pipeline: String,
    val label: String,
    val conf: Float,
    val box: NormBox,
    val trackId: String? = null,
    val seenMsAgo: Int = 0,
    /** Secondary text: match score, raw OCR text… */
    val detail: String? = null,
    val points: List<KeyPoint> = emptyList(),
)

data class NormBox(val x1: Float, val y1: Float, val x2: Float, val y2: Float)

/** What the client sends for one pipeline in `config.pipelines`; null [enabled] = leave as is. */
data class PipelineConfig(val enabled: Boolean? = null, val options: Map<String, OptionValue> = emptyMap())

/** Server → client messages (docs/PROTOCOL.md). */
sealed interface ServerMessage {
    data class Hello(
        val protocol: Int,
        val device: String,
        val version: String,
        val pipelines: List<PipelineSpec>,
        /** Optional server capabilities (v3.2: `text_read`); empty for older servers. */
        val features: Set<String> = emptySet(),
    ) : ServerMessage {
        val canReadText: Boolean get() = Protocol.FEATURE_TEXT_READ in features
    }

    data class PipelineStatusChange(
        val pipeline: String,
        val status: PipelineStatus,
        val message: String?,
    ) : ServerMessage

    data class Detections(
        val seq: Long,
        val clientTs: Long,
        val width: Int,
        val height: Int,
        val decodeMs: Double,
        val serverMs: Double,
        /** Inference time per pipeline that ran on this frame, in server order. */
        val pipelineMs: Map<String, Double>,
        /** Raw annotations of the pipelines that ran on this frame. */
        val dets: List<Detection>,
        /** Server-side smoothed tracks of all pipelines (ByteTrack + hold + label vote). */
        val tracks: List<Detection>,
    ) : ServerMessage

    data class CommandResult(
        val req: Long,
        val pipeline: String,
        val action: String,
        val ok: Boolean,
        val message: String,
        val items: List<ResultItem>,
        val itemAction: ItemAction?,
        /** Files the action produced on the server (v3.1), downloaded over HTTP into the gallery. */
        val files: List<ResultFile> = emptyList(),
    ) : ServerMessage {
        data class ResultItem(val key: String, val label: String)
        data class ItemAction(val id: String, val label: String, val confirm: Boolean)

        /** [url] is server-relative (`/captures/<id>/<name>`); [kind] is informational (photo, openpose, json…). */
        data class ResultFile(val name: String, val kind: String, val url: String)
    }

    data class Dropped(val seq: Long) : ServerMessage

    data class Error(val seq: Long?, val message: String) : ServerMessage
}

/** Wire format v3: binary frame header + JSON parsing/serialization. Unknown fields are ignored. */
object Protocol {
    const val VERSION = 3
    const val HEADER_SIZE = 16
    /** `hello.features` entry: the server answers `POST /text/read` (docs/PROTOCOL.md "Read text"). */
    const val FEATURE_TEXT_READ = "text_read"

    /** 16-byte big-endian header `[u32 seq, i64 client_ts ms, u32 reserved=0]` followed by the JPEG. */
    fun encodeFrame(seq: Long, clientTsMs: Long, jpeg: ByteArray, jpegLength: Int = jpeg.size): ByteArray {
        val out = ByteArray(HEADER_SIZE + jpegLength)
        ByteBuffer.wrap(out).order(ByteOrder.BIG_ENDIAN).apply {
            putInt((seq and 0xFFFF_FFFFL).toInt())
            putLong(clientTsMs)
            putInt(0)
        }
        System.arraycopy(jpeg, 0, out, HEADER_SIZE, jpegLength)
        return out
    }

    /** `config` message; null fields are omitted (= server keeps its current value). */
    fun configJson(
        conf: Float? = null,
        imgsz: Int? = null,
        holdMs: Int? = null,
        pipelines: Map<String, PipelineConfig>? = null,
    ): String {
        // `protocol` tells the server this client speaks v3 (it answers v2-shaped results to older apps).
        val o = JSONObject().put("type", "config").put("protocol", VERSION)
        conf?.let { o.put("conf", it.toDouble()) }
        imgsz?.let { o.put("imgsz", it) }
        holdMs?.let { o.put("hold_ms", it) }
        if (!pipelines.isNullOrEmpty()) {
            val p = JSONObject()
            for ((id, cfg) in pipelines) {
                val c = JSONObject()
                cfg.enabled?.let { c.put("enabled", it) }
                if (cfg.options.isNotEmpty()) c.put("options", valuesJson(cfg.options))
                p.put(id, c)
            }
            o.put("pipelines", p)
        }
        return o.toString()
    }

    fun commandJson(req: Long, pipeline: String, action: String, args: Map<String, OptionValue>): String =
        JSONObject()
            .put("type", "command")
            .put("req", req)
            .put("pipeline", pipeline)
            .put("action", action)
            .put("args", valuesJson(args))
            .toString()

    fun valuesJson(values: Map<String, OptionValue>): JSONObject =
        JSONObject().also { o -> values.forEach { (k, v) -> o.put(k, v.toJsonValue()) } }

    /** Typed value from JSON, or null if it is none of bool/number/string. */
    fun optionValueOf(raw: Any?): OptionValue? = when (raw) {
        is Boolean -> OptionValue.Bool(raw)
        is Number -> OptionValue.Num(raw.toDouble())
        is String -> OptionValue.Str(raw)
        else -> null
    }

    /**
     * The server defaults the text pipeline's `live` option to "Read live" so installed apps (which echo every
     * default back) keep their behaviour; this app has the Read button, so its default is marking text areas.
     * A user's explicit choice still wins (it is an override in [PipelineSettings]).
     */
    private fun markTextByDefault(spec: PipelineSpec): PipelineSpec {
        if (spec.id != LocalCatalog.TEXT) return spec
        return spec.copy(
            options = spec.options.map { option ->
                if (option.key == LocalCatalog.OPT_LIVE && LocalCatalog.LIVE_MARK in option.choices) {
                    option.copy(default = OptionValue.Str(LocalCatalog.LIVE_MARK))
                } else {
                    option
                }
            },
        )
    }

    /** Returns null for malformed or unknown messages. */
    fun parse(text: String): ServerMessage? = try {
        val o = JSONObject(text)
        when (o.optString("type")) {
            "hello" -> {
                val features = o.optJSONArray("features")
                    ?.let { arr -> (0 until arr.length()).mapNotNull { arr.opt(it) as? String }.toSet() }
                    .orEmpty()
                ServerMessage.Hello(
                    protocol = o.optInt("protocol", 0),
                    device = o.optString("device"),
                    version = o.optString("version"),
                    pipelines = o.optJSONArray("pipelines").objects().mapNotNull(::parsePipeline)
                        .map { if (FEATURE_TEXT_READ in features) markTextByDefault(it) else it },
                    features = features,
                )
            }
            "pipeline_status" -> ServerMessage.PipelineStatusChange(
                pipeline = o.getString("pipeline"),
                status = PipelineStatus.fromWire(o.optString("status")),
                message = o.optStringOrNull("message"),
            )
            "detections" -> parseDetections(o)
            "command_result" -> parseCommandResult(o)
            "dropped" -> ServerMessage.Dropped(o.getLong("seq"))
            "error" -> ServerMessage.Error(
                seq = if (o.has("seq") && !o.isNull("seq")) o.getLong("seq") else null,
                message = o.optString("message"),
            )
            else -> null
        }
    } catch (_: JSONException) {
        null
    }

    private fun parsePipeline(o: JSONObject): PipelineSpec? {
        val id = o.optString("id").ifEmpty { return null }
        return PipelineSpec(
            id = id,
            name = o.optString("name").ifEmpty { id },
            description = o.optString("description"),
            defaultEnabled = o.optBoolean("default_enabled", false),
            usesGlobalConf = o.optBoolean("uses_global_conf", false),
            intervalMs = o.optInt("interval_ms", 0),
            options = o.optJSONArray("options").objects().mapNotNull(::parseOption),
            actions = o.optJSONArray("actions").objects().mapNotNull(::parseAction),
            skeleton = parseSkeleton(o.optJSONArray("skeleton")),
            status = PipelineStatus.fromWire(o.optString("status")),
            statusMessage = o.optStringOrNull("message"),
        )
    }

    private fun parseAction(a: JSONObject): ActionSpec? {
        val id = a.optString("id").ifEmpty { return null }
        return ActionSpec(
            id = id,
            label = a.optString("label").ifEmpty { id },
            args = a.optJSONArray("args").objects().mapNotNull(::parseOption),
        )
    }

    /** Null for options without a key or of an unknown type. */
    private fun parseOption(o: JSONObject): OptionSpec? {
        val key = o.optString("key").ifEmpty { return null }
        val type = OptionType.fromWire(o.optString("type")) ?: return null
        val choices = o.optJSONArray("choices")?.let { arr -> List(arr.length()) { arr.optString(it) } } ?: emptyList()
        val fallback: OptionValue = when (type) {
            OptionType.BOOL -> OptionValue.Bool(false)
            OptionType.TEXT -> OptionValue.Str("")
            OptionType.CHOICE -> OptionValue.Str(choices.firstOrNull() ?: "")
            OptionType.NUMBER -> OptionValue.Num(o.optDoubleOrNull("min") ?: 0.0)
        }
        val spec = OptionSpec(
            key = key,
            label = o.optString("label").ifEmpty { key },
            type = type,
            default = fallback,
            min = o.optDoubleOrNull("min"),
            max = o.optDoubleOrNull("max"),
            step = o.optDoubleOrNull("step"),
            choices = choices,
        )
        // A default of the wrong type falls back to a type-correct one.
        return spec.copy(default = spec.coerce(optionValueOf(o.opt("default"))))
    }

    private fun parseSkeleton(arr: JSONArray?): List<Pair<Int, Int>> {
        arr ?: return emptyList()
        return (0 until arr.length()).mapNotNull { i ->
            val e = arr.optJSONArray(i)
            if (e == null || e.length() < 2) return@mapNotNull null
            val a = e.optInt(0, -1)
            val b = e.optInt(1, -1)
            if (a >= 0 && b >= 0) Pair(a, b) else null
        }
    }

    private fun parseDetectionList(o: JSONObject, key: String): List<Detection> =
        o.optJSONArray(key).objects().mapNotNull(::parseDetection)

    private fun parseDetection(d: JSONObject): Detection? {
        val b = d.optJSONArray("box") ?: return null
        if (b.length() < 4) return null
        return Detection(
            pipeline = d.optString("pipeline"),
            label = d.optString("label"),
            conf = d.optDouble("conf", 0.0).toFloat(),
            box = NormBox(
                b.getDouble(0).toFloat(),
                b.getDouble(1).toFloat(),
                b.getDouble(2).toFloat(),
                b.getDouble(3).toFloat(),
            ),
            // v2 servers sent integer ids; they keep working as strings.
            trackId = if (d.has("id") && !d.isNull("id")) d.get("id").toString() else null,
            seenMsAgo = d.optInt("seen_ms_ago", 0),
            detail = d.optStringOrNull("detail"),
            points = parsePoints(d.optJSONArray("points")),
        )
    }

    private fun parsePoints(arr: JSONArray?): List<KeyPoint> {
        arr ?: return emptyList()
        return List(arr.length()) { i ->
            val p = arr.optJSONArray(i)
            if (p == null || p.length() < 2) {
                KeyPoint(0f, 0f, 0f) // keeps indices aligned with the skeleton; invisible
            } else {
                KeyPoint(p.optDouble(0, 0.0).toFloat(), p.optDouble(1, 0.0).toFloat(), p.optDouble(2, 1.0).toFloat())
            }
        }
    }

    private fun parseDetections(o: JSONObject): ServerMessage.Detections {
        val ms = LinkedHashMap<String, Double>()
        o.optJSONObject("pipeline_ms")?.let { p -> p.keys().forEach { k -> ms[k] = p.optDouble(k, 0.0) } }
        return ServerMessage.Detections(
            seq = o.getLong("seq"),
            clientTs = o.getLong("client_ts"),
            width = o.getInt("w"),
            height = o.getInt("h"),
            decodeMs = o.optDouble("decode_ms", 0.0),
            serverMs = o.optDouble("server_ms", 0.0),
            pipelineMs = ms,
            dets = parseDetectionList(o, "dets"),
            tracks = parseDetectionList(o, "tracks"),
        )
    }

    private fun parseCommandResult(o: JSONObject): ServerMessage.CommandResult =
        ServerMessage.CommandResult(
            req = o.optLong("req", -1),
            pipeline = o.optString("pipeline"),
            action = o.optString("action"),
            ok = o.optBoolean("ok", false),
            message = o.optString("message"),
            items = o.optJSONArray("items").objects().mapNotNull { i ->
                val key = i.optString("key")
                if (key.isEmpty()) null else ServerMessage.CommandResult.ResultItem(key, i.optString("label").ifEmpty { key })
            },
            itemAction = o.optJSONObject("item_action")?.let { a ->
                val id = a.optString("id")
                if (id.isEmpty()) null
                else ServerMessage.CommandResult.ItemAction(id, a.optString("label").ifEmpty { id }, a.optBoolean("confirm", false))
            },
            files = o.optJSONArray("files").objects().mapNotNull(::parseResultFile),
        )

    /** Null without a url; a missing name falls back to the url's last path segment. */
    private fun parseResultFile(f: JSONObject): ServerMessage.CommandResult.ResultFile? {
        val url = f.optString("url").trim().ifEmpty { return null }
        val name = f.optString("name").trim().ifEmpty { url.substringBefore('?').substringAfterLast('/') }
        if (name.isEmpty()) return null
        return ServerMessage.CommandResult.ResultFile(name, f.optString("kind"), url)
    }

    private fun JSONArray?.objects(): List<JSONObject> =
        if (this == null) emptyList() else (0 until length()).mapNotNull { optJSONObject(it) }

    private fun JSONObject.optStringOrNull(key: String): String? =
        if (has(key) && !isNull(key)) optString(key).takeIf { it.isNotEmpty() } else null

    private fun JSONObject.optDoubleOrNull(key: String): Double? =
        if (has(key) && !isNull(key)) optDouble(key).takeIf { !it.isNaN() } else null
}
