package pl.genor.glassvision

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ProtocolTest {

    @Test
    fun header_is_16_bytes_big_endian() {
        val jpeg = byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 0x01, 0x02)
        val out = Protocol.encodeFrame(seq = 0x01020304L, clientTsMs = 0x0A0B0C0D0E0F1011L, jpeg = jpeg)

        assertEquals(Protocol.HEADER_SIZE + jpeg.size, out.size)
        val expectedHeader = byteArrayOf(
            0x01, 0x02, 0x03, 0x04,
            0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11,
            0x00, 0x00, 0x00, 0x00,
        )
        assertArrayEquals(expectedHeader, out.copyOfRange(0, 16))
        assertArrayEquals(jpeg, out.copyOfRange(16, out.size))
    }

    @Test
    fun header_round_trips_realistic_values_and_respects_length() {
        val ts = 1_792_681_655_531L
        val buf = ByteArray(64) { it.toByte() }
        val out = Protocol.encodeFrame(seq = 0xFFFF_FFFFL, clientTsMs = ts, jpeg = buf, jpegLength = 10)

        assertEquals(26, out.size)
        val bb = ByteBuffer.wrap(out).order(ByteOrder.BIG_ENDIAN)
        assertEquals(0xFFFF_FFFFL, bb.getInt().toLong() and 0xFFFF_FFFFL)
        assertEquals(ts, bb.getLong())
        assertEquals(0, bb.getInt())
        assertArrayEquals(buf.copyOfRange(0, 10), out.copyOfRange(16, 26))
    }

    @Test
    fun parses_v3_hello_with_pipelines_options_actions_skeleton() {
        val msg = Protocol.parse(HELLO_V3) as ServerMessage.Hello
        assertEquals(3, msg.protocol)
        assertEquals("0.4.0.3", msg.version)
        assertEquals("cuda:0 (NVIDIA GeForce RTX 3090)", msg.device)
        assertEquals(listOf("objects", "open_vocab", "faces", "gestures", "text"), msg.pipelines.map { it.id })

        val objects = msg.pipelines[0]
        assertTrue(objects.defaultEnabled)
        assertTrue(objects.usesGlobalConf)
        assertEquals(PipelineStatus.READY, objects.status)
        assertTrue(objects.options.isEmpty() && objects.actions.isEmpty() && objects.skeleton.isEmpty())

        val openVocab = msg.pipelines[1]
        assertEquals(
            OptionSpec("prompt", "What to look for (comma-separated)", OptionType.TEXT, OptionValue.Str("person, cup")),
            openVocab.options.single(),
        )
        assertEquals(PipelineStatus.NOT_LOADED, openVocab.status)

        val faces = msg.pipelines[2]
        val threshold = faces.options.single()
        assertEquals(OptionType.NUMBER, threshold.type)
        assertEquals(OptionValue.Num(0.4), threshold.default)
        assertEquals(0.2, threshold.min!!, 1e-9)
        assertEquals(0.8, threshold.max!!, 1e-9)
        assertEquals(0.05, threshold.step!!, 1e-9)
        assertTrue(threshold.isSlider)
        assertEquals(listOf("enroll", "list"), faces.actions.map { it.id })
        assertEquals(
            OptionSpec("name", "Name", OptionType.TEXT, OptionValue.Str("")),
            faces.actions[0].args.single(),
        )
        assertTrue(faces.actions[1].args.isEmpty())

        val gestures = msg.pipelines[3]
        assertEquals(listOf(0 to 1, 1 to 2, 2 to 3), gestures.skeleton)
        assertEquals("Gestures", gestures.name)
        assertEquals(PipelineStatus.ERROR, gestures.status)
        assertEquals("CUDA out of memory", gestures.statusMessage)
        // Options of unknown types are skipped; missing name falls back to the id.
        val lang = gestures.options.single()
        assertEquals(OptionType.CHOICE, lang.type)
        assertEquals(listOf("latin", "cyrillic"), lang.choices)
        // Default not among the choices → first choice.
        assertEquals(OptionValue.Str("latin"), lang.default)

        val text = msg.pipelines[4]
        assertEquals("text", text.name)
        assertEquals(500, text.intervalMs)
        assertEquals(PipelineStatus.UNKNOWN, text.status)
        assertEquals(OptionSpec("show_raw", "Show raw", OptionType.BOOL, OptionValue.Bool(true)), text.options.single())
    }

    @Test
    fun parses_pipeline_status() {
        assertEquals(
            ServerMessage.PipelineStatusChange("text", PipelineStatus.ERROR, "CUDA out of memory"),
            Protocol.parse("""{"type":"pipeline_status","pipeline":"text","status":"error","message":"CUDA out of memory"}"""),
        )
        assertEquals(
            ServerMessage.PipelineStatusChange("faces", PipelineStatus.LOADING, null),
            Protocol.parse("""{"type":"pipeline_status","pipeline":"faces","status":"loading"}"""),
        )
        assertNull(Protocol.parse("""{"type":"pipeline_status","status":"ready"}"""))
    }

    @Test
    fun parses_v3_detections_with_string_ids_points_detail_and_pipeline_ms() {
        val msg = Protocol.parse(
            """{"type":"detections","seq":42,"client_ts":1792681655531,"w":960,"h":720,
               "decode_ms":2.1,"server_ms":14.0,"pipeline_ms":{"objects":8.1,"faces":4.2,"text":120},
               "future_field":{"x":1},
               "dets":[{"pipeline":"objects","label":"person","conf":0.91,"box":[0.12,0.30,0.48,0.97]}],
               "tracks":[
                 {"id":"objects:7","pipeline":"objects","label":"person","conf":0.88,"box":[0.1,0.3,0.5,0.9],"seen_ms_ago":120},
                 {"id":"faces:2","pipeline":"faces","label":"Alex","detail":"0.71","conf":0.93,"box":[0.2,0.31,0.33,0.5],"seen_ms_ago":0},
                 {"id":"gestures:1","pipeline":"gestures","label":"Thumb_Up","conf":0.87,"box":[0.6,0.4,0.8,0.7],
                  "points":[[0.71,0.66,1.0],[0.72,0.6,0.2],[0.7,0.5]]},
                 {"id":"bad","pipeline":"objects","label":"no box"}]}"""
        ) as ServerMessage.Detections
        assertEquals(42L, msg.seq)
        assertEquals(1792681655531L, msg.clientTs)
        assertEquals(960, msg.width)
        assertEquals(720, msg.height)
        assertEquals(14.0, msg.serverMs, 1e-9)
        // Key order follows the JSON implementation (Android keeps it, JVM org.json doesn't): compare as a set.
        assertEquals(setOf("objects", "faces", "text"), msg.pipelineMs.keys)
        assertEquals(120.0, msg.pipelineMs.getValue("text"), 1e-9)
        assertEquals(Detection("objects", "person", 0.91f, NormBox(0.12f, 0.30f, 0.48f, 0.97f)), msg.dets.single())

        assertEquals(3, msg.tracks.size) // the box-less entry is dropped
        assertEquals(
            Detection("objects", "person", 0.88f, NormBox(0.1f, 0.3f, 0.5f, 0.9f), trackId = "objects:7", seenMsAgo = 120),
            msg.tracks[0],
        )
        assertEquals("faces:2", msg.tracks[1].trackId)
        assertEquals("0.71", msg.tracks[1].detail)
        assertEquals(
            listOf(KeyPoint(0.71f, 0.66f, 1f), KeyPoint(0.72f, 0.6f, 0.2f), KeyPoint(0.7f, 0.5f, 1f)),
            msg.tracks[2].points,
        )
        assertNull(msg.tracks[2].detail)
    }

    @Test
    fun v2_integer_track_ids_and_missing_fields_still_parse() {
        val msg = Protocol.parse(
            """{"type":"detections","seq":1,"client_ts":2,"w":4,"h":3,"infer_ms":6.4,
               "tracks":[{"id":7,"cls":0,"label":"person","conf":0.5,"box":[0,0,1,1]}]}"""
        ) as ServerMessage.Detections
        assertEquals("7", msg.tracks.single().trackId)
        assertEquals("", msg.tracks.single().pipeline)
        assertTrue(msg.dets.isEmpty())
        assertTrue(msg.pipelineMs.isEmpty())
    }

    @Test
    fun parses_command_result_with_items_and_item_action() {
        val msg = Protocol.parse(
            """{"type":"command_result","req":7,"pipeline":"faces","action":"list","ok":true,
               "message":"2 known faces",
               "items":[{"key":"Alex","label":"Alex (3 samples)"},{"key":"Sam"},{"label":"no key"}],
               "item_action":{"id":"delete","label":"Forget","confirm":true}}"""
        ) as ServerMessage.CommandResult
        assertEquals(7L, msg.req)
        assertEquals("faces", msg.pipeline)
        assertEquals("list", msg.action)
        assertTrue(msg.ok)
        assertEquals("2 known faces", msg.message)
        assertEquals(
            listOf(
                ServerMessage.CommandResult.ResultItem("Alex", "Alex (3 samples)"),
                ServerMessage.CommandResult.ResultItem("Sam", "Sam"),
            ),
            msg.items,
        )
        assertEquals(ServerMessage.CommandResult.ItemAction("delete", "Forget", true), msg.itemAction)

        val plain = Protocol.parse("""{"type":"command_result","req":8,"ok":false,"message":"no face in frame"}""")
            as ServerMessage.CommandResult
        assertFalse(plain.ok)
        assertTrue(plain.items.isEmpty())
        assertNull(plain.itemAction)
        assertTrue(plain.files.isEmpty())
    }

    @Test
    fun parses_command_result_files() {
        val msg = Protocol.parse(
            """{"type":"command_result","req":9,"pipeline":"pose","action":"capture","ok":true,
               "message":"Captured 2 people → 4 files",
               "files":[{"name":"photo.jpg","kind":"photo","url":"/captures/20261023-101500-123/photo.jpg"},
                        {"name":"openpose_body.png","kind":"openpose","url":"/captures/20261023-101500-123/openpose_body.png"},
                        {"kind":"json","url":"/captures/20261023-101500-123/pose_keypoints.json"},
                        {"name":"no-url.png","kind":"openpose"},
                        "not an object"]}"""
        ) as ServerMessage.CommandResult
        assertEquals(
            listOf(
                ServerMessage.CommandResult.ResultFile("photo.jpg", "photo", "/captures/20261023-101500-123/photo.jpg"),
                ServerMessage.CommandResult.ResultFile("openpose_body.png", "openpose", "/captures/20261023-101500-123/openpose_body.png"),
                // Missing name: the url's last segment.
                ServerMessage.CommandResult.ResultFile("pose_keypoints.json", "json", "/captures/20261023-101500-123/pose_keypoints.json"),
            ),
            msg.files,
        )
        assertTrue(msg.items.isEmpty())
    }

    @Test
    fun parses_control_messages_and_rejects_garbage() {
        assertEquals(ServerMessage.Dropped(41), Protocol.parse("""{"type":"dropped","seq":41}"""))
        assertEquals(
            ServerMessage.Error(43, "cannot decode JPEG"),
            Protocol.parse("""{"type":"error","seq":43,"message":"cannot decode JPEG"}"""),
        )
        assertEquals(ServerMessage.Error(null, "bad config"), Protocol.parse("""{"type":"error","message":"bad config"}"""))
        assertNull(Protocol.parse("not json"))
        assertNull(Protocol.parse("""{"type":"mystery"}"""))
        // A v2 hello (no pipelines) parses to an empty pipeline list.
        assertEquals(ServerMessage.Hello(2, "cuda:0", "", emptyList()), Protocol.parse("""{"type":"hello","protocol":2,"device":"cuda:0","classes":["a"]}"""))
    }

    @Test
    fun config_json_globals_and_pipelines() {
        val o = JSONObject(
            Protocol.configJson(
                0.35f, 640, 300,
                linkedMapOf(
                    "objects" to PipelineConfig(enabled = true),
                    "open_vocab" to PipelineConfig(true, mapOf("prompt" to OptionValue.Str("red mug, keys"))),
                    "faces" to PipelineConfig(null, mapOf("threshold" to OptionValue.Num(0.45), "max" to OptionValue.Num(3.0))),
                ),
            )
        )
        assertEquals("config", o.getString("type"))
        assertEquals(0.35, o.getDouble("conf"), 1e-6)
        assertEquals(640, o.getInt("imgsz"))
        assertEquals(300, o.getInt("hold_ms"))
        val p = o.getJSONObject("pipelines")
        assertEquals(true, p.getJSONObject("objects").getBoolean("enabled"))
        assertFalse(p.getJSONObject("objects").has("options"))
        assertEquals("red mug, keys", p.getJSONObject("open_vocab").getJSONObject("options").getString("prompt"))
        val faces = p.getJSONObject("faces")
        assertFalse(faces.has("enabled"))
        assertEquals(0.45, faces.getJSONObject("options").getDouble("threshold"), 1e-9)
        // Whole numbers go out as integers.
        assertEquals(3, faces.getJSONObject("options").get("max"))
    }

    @Test
    fun config_json_omits_unset_fields() {
        val o = JSONObject(Protocol.configJson(pipelines = mapOf("text" to PipelineConfig(enabled = false))))
        assertEquals(setOf("type", "protocol", "pipelines"), o.keySet())
        assertFalse(o.getJSONObject("pipelines").getJSONObject("text").getBoolean("enabled"))
        assertEquals(setOf("type", "protocol", "conf", "imgsz", "hold_ms"), JSONObject(Protocol.configJson(0.5f, 320, 0)).keySet())
    }

    @Test
    fun command_json_matches_protocol() {
        val o = JSONObject(Protocol.commandJson(7, "faces", "enroll", mapOf("name" to OptionValue.Str("Alex"))))
        assertEquals("command", o.getString("type"))
        assertEquals(7L, o.getLong("req"))
        assertEquals("faces", o.getString("pipeline"))
        assertEquals("enroll", o.getString("action"))
        assertEquals("Alex", o.getJSONObject("args").getString("name"))
        assertEquals(0, JSONObject(Protocol.commandJson(8, "faces", "list", emptyMap())).getJSONObject("args").length())
    }

    companion object {
        val HELLO_V3 = """
            {"type":"hello","protocol":3,"device":"cuda:0 (NVIDIA GeForce RTX 3090)","version":"0.4.0.3","extra":42,
             "pipelines":[
              {"id":"objects","name":"Objects (80 COCO classes)","description":"YOLO11s","default_enabled":true,
               "uses_global_conf":true,"interval_ms":0,"options":[],"actions":[],"skeleton":null,"status":"ready"},
              {"id":"open_vocab","name":"Find by text","default_enabled":false,"uses_global_conf":true,
               "options":[{"key":"prompt","label":"What to look for (comma-separated)","type":"text","default":"person, cup"}],
               "actions":[],"skeleton":null,"status":"not_loaded"},
              {"id":"faces","name":"Faces","options":[{"key":"threshold","label":"Match threshold","type":"number",
                "default":0.4,"min":0.2,"max":0.8,"step":0.05}],
               "actions":[{"id":"enroll","label":"Learn face","args":[{"key":"name","label":"Name","type":"text","default":""}]},
                          {"id":"list","label":"Known faces","args":[]}],
               "skeleton":null,"status":"not_loaded"},
              {"id":"gestures","name":"Gestures","skeleton":[[0,1],[1,2],[2,3],[4]],"status":"error","message":"CUDA out of memory",
               "options":[{"key":"lang","label":"Script","type":"choice","default":"greek","choices":["latin","cyrillic"]},
                          {"key":"color","label":"Colour","type":"rgb","default":"#fff"}]},
              {"id":"text","interval_ms":500,"status":"warming_up",
               "options":[{"key":"show_raw","label":"Show raw","type":"bool","default":true}]},
              {"name":"no id"}
             ]}
        """.trimIndent()
    }
}
