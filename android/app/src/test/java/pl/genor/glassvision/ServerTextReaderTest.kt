package pl.genor.glassvision

import okhttp3.Interceptor
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import okio.Buffer
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/** docs/PROTOCOL.md "Read text": the JSON of the three endpoints, the requests, errors, and `hello.features`. */
class ServerTextReaderTest {

    /** The response example from docs/PROTOCOL.md, verbatim. */
    private val protocolExample = """
        {"w": 4000, "h": 3000, "script": "latin", "ms": {"ocr": 310.2, "translate": 120.5},
         "lines": [{"text": "Ausgang", "conf": 0.93, "box": [0.12, 0.30, 0.41, 0.38], "block": 0, "lang": "de"},
                   {"text": "Notausgang 2", "conf": 0.88, "box": [0.12, 0.40, 0.55, 0.47], "block": 0, "lang": "de"}],
         "translations": {"pl": ["Wyjście", "Wyjście awaryjne 2"]},
         "errors": {}}
    """.trimIndent()

    @Test
    fun parses_the_protocol_example() {
        val r = TextReadJson.reading(JSONObject(protocolExample))
        assertEquals(4000, r.width)
        assertEquals(3000, r.height)
        assertEquals("latin", r.script)
        assertEquals(listOf("Ausgang", "Notausgang 2"), r.lines.map { it.text })
        assertEquals(listOf(0, 0), r.lines.map { it.block })
        assertEquals(listOf("de", "de"), r.lines.map { it.lang })
        assertEquals(0.93f, r.lines[0].conf, 1e-6f)
        val box = r.lines[1].box
        assertEquals(listOf(0.12f, 0.40f, 0.55f, 0.47f), listOf(box.left, box.top, box.right, box.bottom))
        // v3.2: one block from the lines' `block`, its translation from theirs.
        assertEquals(listOf(listOf(0, 1)), r.blocks.map { it.lines })
        assertEquals(listOf("Wyjście\nWyjście awaryjne 2"), r.translation("pl"))
        assertEquals(0f, r.angle)
        assertTrue(r.errors.isEmpty())
    }

    @Test
    fun null_entries_failed_targets_and_unknown_languages() {
        val r = TextReadJson.reading(
            JSONObject(
                """{"w": 800, "h": 600, "script": null,
                    "lines": [{"text": "Bahnhof", "conf": 0.9, "box": [0,0,1,1], "block": 0, "lang": null},
                              {"text": "12:45", "conf": 0.8, "box": [0,0,1,1], "block": 1, "lang": "de"},
                              {"text": "Gleis 3", "conf": 0.7, "box": [0,0,1,1], "block": 1}],
                    "translations": {"en": ["Station", null], "pl": [null, null, "Peron 3", "extra"]},
                    "errors": {"xx": "unknown language"}}"""
            )
        )
        assertNull(r.script)
        assertEquals(listOf(null, "de", null), r.lines.map { it.lang })
        // Short arrays are padded with null, long ones cut to the lines; a block keeps untranslated lines as read.
        assertEquals(listOf("Station", null), r.translation("en"))
        assertEquals(listOf(null, "12:45\nPeron 3"), r.translation("pl"))
        assertEquals(mapOf("xx" to "unknown language"), r.errors)
    }

    @Test
    fun a_skipped_line_keeps_the_translations_aligned() {
        val r = TextReadJson.reading(
            JSONObject(
                """{"w": 1, "h": 1, "lines": [{"text": "A", "box": [0,0,1,1], "block": 0}, {"text": "  "},
                    "junk", {"text": "C", "box": [0,0,1,1], "block": 2}],
                    "translations": {"pl": ["a", "blank", "junk", "c"]}}"""
            )
        )
        assertEquals(listOf("A", "C"), r.lines.map { it.text })
        assertEquals(listOf("a", "c"), r.translation("pl"))
    }

    @Test
    fun no_text_found() {
        val r = TextReadJson.reading(JSONObject("""{"w": 4000, "h": 3000, "script": "latin", "lines": [], "translations": {"pl": []}, "errors": {}}"""))
        assertTrue(r.lines.isEmpty())
        assertTrue(r.blocks.isEmpty())
    }

    @Test
    fun missing_block_falls_back_to_one_block_per_line() {
        val r = TextReadJson.reading(JSONObject("""{"lines": [{"text": "A"}, {"text": "B", "block": null}]}"""))
        assertEquals(listOf(0, 1), r.lines.map { it.block })
        assertEquals(listOf(0, 1), r.blocks.map { it.id })
        assertEquals(0, r.width)
    }

    /** The v3.3 example from docs/PROTOCOL.md "Read text v3.3", with its placeholders filled in. */
    private val blocksExample = """
        {"w": 4000, "h": 3000, "script": "latin", "angle": 90.0, "ms": {"ocr": 310.2, "translate": 120.5},
         "lines": [{"text": "Ausgang", "conf": 0.93, "box": [0.12, 0.30, 0.41, 0.38],
                    "quad": [[0.41, 0.30], [0.41, 0.38], [0.12, 0.38], [0.12, 0.30]],
                    "block": 0, "lang": "de", "script": "latin", "suspect": false},
                   {"text": "Notausgang 2", "conf": 0.88, "box": [0.12, 0.40, 0.55, 0.47],
                    "quad": [[0.55, 0.40], [0.55, 0.47], [0.12, 0.47], [0.12, 0.40]],
                    "block": 0, "lang": "de", "script": "latin", "suspect": false},
                   {"text": "BbIXOI", "conf": 0.41, "box": [0.60, 0.70, 0.80, 0.75], "block": 1, "lang": null,
                    "script": "latin", "suspect": true}],
         "blocks": [{"id": 0, "lines": [0, 1], "box": [0.12, 0.30, 0.55, 0.47],
                     "quad": [[0.55, 0.30], [0.55, 0.47], [0.12, 0.47], [0.12, 0.30]],
                     "text": "Ausgang\nNotausgang 2", "kind": "list", "lang": "de", "suspect": false,
                     "translations": {"pl": "Wyjście\nWyjście awaryjne 2", "en": "Exit\nEmergency exit 2"}},
                    {"id": 1, "lines": [2], "box": [0.60, 0.70, 0.80, 0.75], "text": "BbIXOI", "kind": "prose",
                     "lang": null, "suspect": true, "translations": {"pl": null, "en": null}}],
         "translations": {},
         "errors": {"ja": "model not installed"}}
    """.trimIndent()

    @Test
    fun parses_the_v33_blocks_example() {
        val r = TextReadJson.reading(JSONObject(blocksExample))
        assertEquals(90f, r.angle)
        assertEquals(listOf("latin", "latin", "latin"), r.lines.map { it.script })
        assertEquals(listOf(false, false, true), r.lines.map { it.suspect })
        assertEquals(Pt(0.41f, 0.30f), r.lines[0].quad!!.points[0])
        assertNull(r.lines[2].quad)
        assertEquals(2, r.blocks.size)
        val (sign, garbled) = r.blocks
        assertEquals(listOf(0, 1), sign.lines)
        assertEquals(TextBox(0.12f, 0.30f, 0.55f, 0.47f), sign.box)
        assertEquals(90f, sign.quad!!.angle) // rotated text: the quad runs downwards
        assertEquals("Ausgang\nNotausgang 2", sign.text)
        assertEquals(BlockKind.LIST, sign.kind)
        assertEquals("de", sign.lang)
        assertEquals(mapOf("pl" to "Wyjście\nWyjście awaryjne 2", "en" to "Exit\nEmergency exit 2"), sign.translations)
        assertTrue(garbled.suspect)
        assertEquals(BlockKind.PROSE, garbled.kind)
        assertNull(garbled.lang)
        assertEquals(listOf("Wyjście\nWyjście awaryjne 2", null), r.translation("pl"))
        assertEquals(setOf("pl", "en"), r.translated)
        assertEquals(mapOf("ja" to "model not installed"), r.errors)
    }

    @Test
    fun v33_blocks_survive_skipped_lines_and_missing_fields() {
        val r = TextReadJson.reading(
            JSONObject(
                """{"w": 1, "h": 1, "lines": [{"text": " "}, {"text": "A", "box": [0.1, 0.1, 0.2, 0.2], "block": 0},
                    {"text": "B", "box": [0.1, 0.3, 0.4, 0.4], "block": 0}],
                    "blocks": [{"lines": [0, 1, 2, 99, "x"], "quad": [[0, 0], [1, 0]]}, {"id": 5, "lines": [0]}, "junk"]}"""
            )
        )
        // Line 0 was blank: block members are remapped past it, the block made only of it is dropped.
        val b = r.blocks.single()
        assertEquals(listOf(0, 1), b.lines)
        assertEquals(0, b.id)
        assertEquals("A\nB", b.text) // no `text`: the lines'
        assertEquals(TextBox(0.1f, 0.1f, 0.4f, 0.4f), b.box) // no `box`: the lines' union
        assertNull(b.quad) // a quad must have 4 points
        assertEquals(BlockKind.LIST, b.kind) // no `kind`: guessed (one-word lines)
        assertFalse(b.suspect)
        assertTrue(r.translated.isEmpty())
    }

    @Test
    fun translate_and_languages_answers() {
        assertEquals(
            listOf("Sortie", null, null),
            TextReadJson.translations(JSONObject("""{"to": "fr", "translations": ["Sortie", null]}"""), 3),
        )
        val langs = TextReadJson.languages(
            JSONObject("""{"languages": [{"code": "pl", "name": "Polish"}, {"code": "de", "name": "German"},
                {"code": "pl", "name": "Polish again"}, {"name": "No code"}, {"code": "xx"}]}""")
        )
        assertEquals(listOf("German", "Polish", "xx"), langs.map { it.name })
    }

    @Test
    fun error_messages() {
        assertEquals("OCR model could not be loaded: CUDA OOM (HTTP 503)", TextReadJson.errorMessage(503, """{"detail": "OCR model could not be loaded: CUDA OOM"}"""))
        assertEquals("Server error (HTTP 400)", TextReadJson.errorMessage(400, """{"detail": [{"loc": ["query"], "msg": "bad"}]}"""))
        assertEquals("Server error (HTTP 500)", TextReadJson.errorMessage(500, "Internal Server Error"))
        assertEquals("Access token rejected (HTTP 401)", TextReadJson.errorMessage(401, """{"detail": "unauthorized"}"""))
        assertEquals("This server cannot read text (HTTP 404)", TextReadJson.errorMessage(404, ""))
    }

    // Requests (OkHttp with a fake server in an interceptor: no network)

    private class FakeServer(val respond: (Request) -> Pair<Int, String>) : Interceptor {
        val requests = mutableListOf<Request>()

        override fun intercept(chain: Interceptor.Chain): Response {
            val request = chain.request()
            requests += request
            val (code, body) = respond(request)
            return Response.Builder()
                .request(request)
                .protocol(okhttp3.Protocol.HTTP_1_1)
                .code(code)
                .message("fake")
                .body(body.toResponseBody("application/json".toMediaType()))
                .build()
        }
    }

    private val endpoint = ServerEndpoint("ws://192.168.1.50:8770/ws", "example-token")
    private val languagesJson = """{"languages": [{"code": "de", "name": "German"}, {"code": "pl", "name": "Polish"}]}"""

    private fun reader(server: FakeServer, options: ReadOptions = ReadOptions()) =
        ServerTextReader(endpoint, { options }, OkHttpClient.Builder().addInterceptor(server).build())

    private fun Request.bodyText(): String = Buffer().also { body!!.writeTo(it) }.readUtf8()

    @Test
    fun read_posts_the_jpeg_with_targets_options_and_token() {
        val server = FakeServer { req ->
            if (req.url.encodedPath == "/text/languages") 200 to languagesJson else 200 to protocolExample
        }
        val options = ReadOptions(languages = "Auto-detect (available scripts)", model = "Fast — M2M100 418M", translateFrom = "German")
        val result = reader(server, options).readJpeg(byteArrayOf(1, 2, 3), listOf("pl", "fr"))

        assertEquals(2, result.lines.size)
        val read = server.requests.single { it.url.encodedPath == "/text/read" }
        assertEquals("POST", read.method)
        assertEquals("image/jpeg", read.body!!.contentType().toString())
        assertEquals(3L, read.body!!.contentLength())
        assertEquals("http", read.url.scheme)
        assertEquals(8770, read.url.port)
        assertEquals("pl,fr", read.url.queryParameter("translate_to"))
        assertEquals("Auto-detect (available scripts)", read.url.queryParameter("languages"))
        assertEquals("Fast — M2M100 418M", read.url.queryParameter("model"))
        // The pipeline option is a language name; the endpoint takes its code.
        assertEquals("de", read.url.queryParameter("translate_from"))
        assertEquals("Bearer example-token", read.header("Authorization"))
        assertNull(read.url.queryParameter("token"))
    }

    @Test
    fun read_without_targets_or_options_sends_no_query() {
        val server = FakeServer { 200 to protocolExample }
        reader(server, ReadOptions(translateFrom = "Auto-detect")).readJpeg(byteArrayOf(1), emptyList())
        val read = server.requests.last()
        assertEquals("/text/read", read.url.encodedPath)
        assertEquals(emptySet<String>(), read.url.queryParameterNames)
    }

    @Test
    fun translate_posts_lines_and_sources() {
        val server = FakeServer { 200 to """{"to": "fr", "translations": ["Sortie", "Sortie de secours 2"]}""" }
        val out = reader(server, ReadOptions(model = "Better")).translate(listOf("Ausgang", "Notausgang 2"), listOf("de", null), "fr")
        assertEquals(listOf("Sortie", "Sortie de secours 2"), out)
        val req = server.requests.single()
        assertEquals("/text/translate", req.url.encodedPath)
        assertEquals("POST", req.method)
        val body = JSONObject(req.bodyText())
        assertEquals("fr", body.getString("to"))
        assertEquals("Better", body.getString("model"))
        assertEquals("Ausgang", body.getJSONArray("lines").getString(0))
        assertEquals("de", body.getJSONArray("from").getString(0))
        assertTrue(body.getJSONArray("from").isNull(1))
    }

    @Test
    fun a_text_blocks_server_is_asked_for_blocks_and_translates_blocks() {
        val server = FakeServer { req ->
            if (req.url.encodedPath == "/text/translate") 200 to """{"to": "fr", "translations": ["Sortie", null]}""" else 200 to blocksExample
        }
        val reader = ServerTextReader(endpoint, { ReadOptions(model = "Better") }, OkHttpClient.Builder().addInterceptor(server).build(), setOf("text_read", "text_blocks"))
        val reading = reader.readJpeg(byteArrayOf(1), listOf("pl"))
        assertEquals("blocks", server.requests.last().url.queryParameter("format"))

        assertEquals(listOf("Sortie", null), reader.translateBlocks(reading.blocks, "fr"))
        val body = JSONObject(server.requests.last().bodyText())
        assertFalse(body.has("lines"))
        assertEquals("fr", body.getString("to"))
        assertEquals("Better", body.getString("model"))
        val blocks = body.getJSONArray("blocks")
        assertEquals(2, blocks.length())
        assertEquals("Ausgang\nNotausgang 2", blocks.getJSONObject(0).getString("text"))
        assertEquals("de", blocks.getJSONObject(0).getString("lang"))
        assertEquals("list", blocks.getJSONObject(0).getString("kind"))
        assertTrue(blocks.getJSONObject(1).isNull("lang"))
        assertEquals("prose", blocks.getJSONObject(1).getString("kind"))
    }

    @Test
    fun an_older_server_gets_no_format_and_line_translation() {
        val server = FakeServer { req ->
            if (req.url.encodedPath == "/text/translate") 200 to """{"to": "fr", "translations": ["Sortie", null, null]}""" else 200 to protocolExample
        }
        val reader = reader(server)
        val reading = reader.readJpeg(byteArrayOf(1), emptyList())
        assertNull(server.requests.last().url.queryParameter("format"))
        val extra = ReadBlock(1, emptyList(), text = "12:45", kind = BlockKind.LIST)
        assertEquals(listOf("Sortie\nNotausgang 2", null), reader.translateBlocks(reading.blocks + extra, "fr"))
        val body = JSONObject(server.requests.last().bodyText())
        assertEquals(3, body.getJSONArray("lines").length())
        assertEquals("de", body.getJSONArray("from").getString(1))
        assertTrue(body.getJSONArray("from").isNull(2))
    }

    @Test
    fun languages_are_fetched_once() {
        val server = FakeServer { 200 to languagesJson }
        val r = reader(server)
        assertEquals(listOf("de", "pl"), r.languages().map { it.code })
        assertEquals(listOf("de", "pl"), r.languages().map { it.code })
        assertEquals(1, server.requests.size)
        assertEquals("GET", server.requests.single().method)
    }

    @Test
    fun http_errors_carry_the_servers_reason() {
        val server = FakeServer { 503 to """{"detail": "OCR model could not be loaded"}""" }
        try {
            reader(server).readJpeg(byteArrayOf(1), emptyList())
            fail("expected TextReadException")
        } catch (e: TextReadException) {
            assertEquals("OCR model could not be loaded (HTTP 503)", e.message)
        }
    }

    @Test
    fun unreachable_server_and_garbage_answers() {
        val down = OkHttpClient.Builder().addInterceptor { throw java.net.ConnectException("refused") }.build()
        try {
            ServerTextReader(endpoint, client = down).languages()
            fail("expected TextReadException")
        } catch (e: TextReadException) {
            assertEquals("Server not reachable: refused", e.message)
        }
        try {
            reader(FakeServer { 200 to "<html>" }).languages()
            fail("expected TextReadException")
        } catch (e: TextReadException) {
            assertEquals("Unexpected answer from the server", e.message)
        }
    }

    // hello.features

    @Test
    fun hello_features() {
        val with = Protocol.parse("""{"type":"hello","protocol":3,"device":"cpu","version":"0.7","pipelines":[],"features":["text_read",7,null]}""")
            as ServerMessage.Hello
        assertEquals(setOf("text_read"), with.features)
        assertTrue(with.canReadText)

        val old = Protocol.parse("""{"type":"hello","protocol":3,"device":"cpu","version":"0.6","pipelines":[]}""") as ServerMessage.Hello
        assertEquals(emptySet<String>(), old.features)
        assertFalse(old.canReadText)

        val other = Protocol.parse("""{"type":"hello","protocol":3,"pipelines":[],"features":["something_else"]}""") as ServerMessage.Hello
        assertFalse(other.canReadText)

        assertFalse(with.canReadBlocks)
        val v33 = Protocol.parse("""{"type":"hello","protocol":3,"pipelines":[],"features":["text_read","text_blocks","text_careful"]}""")
            as ServerMessage.Hello
        assertTrue(v33.canReadBlocks)
        assertTrue(v33.canReadCarefully)
        // Blocks without Read text make no sense: ignored.
        val odd = Protocol.parse("""{"type":"hello","protocol":3,"pipelines":[],"features":["text_blocks"]}""") as ServerMessage.Hello
        assertFalse(odd.canReadBlocks)
    }

    @Test
    fun older_server_hello_without_features_cannot_read_text() {
        // The captured fixture is from a v3.2 server; strip `features` to get what a v3.1 server sends.
        val captured = JSONObject(requireNotNull(javaClass.classLoader?.getResource("hello_v3.json")).readText())
        captured.remove("features")
        val hello = Protocol.parse(captured.toString()) as ServerMessage.Hello
        assertFalse(hello.canReadText)
        // Without the Read button the server's own `live` default (keep reading live) is left alone.
        val live = hello.pipelines.first { it.id == "text" }.options.first { it.key == "live" }
        assertEquals(OptionValue.Str("Read live"), live.default)
    }
}
