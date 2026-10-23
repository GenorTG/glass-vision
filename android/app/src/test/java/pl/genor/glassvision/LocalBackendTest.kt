package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.Executor

/**
 * [LocalBackend] with fake pipelines and synchronous/queued lanes: the messages it emits must have the server's
 * shapes (hello, pipeline_status, detections, command_result), so the rest of the app can't tell the difference.
 */
class LocalBackendTest {
    private class Frame(
        override val width: Int = 640,
        override val height: Int = 480,
        override val capturedAtMs: Long = 1_000,
        override val uptimeMs: Long = 50,
        override val prepMs: Double = 1.5,
    ) : LocalFrame

    private class FakePipeline(
        val id: String,
        var output: (LocalFrame) -> List<PixelDetection> = { emptyList() },
    ) : LocalPipeline {
        val timestamps = mutableListOf<Long>()
        val optionsSeen = mutableListOf<Map<String, OptionValue>>()
        var closed = false
        var failRun: Exception? = null
        var failPrepare: Exception? = null
        var lastCommandFrame: LocalFrame? = null

        override fun run(frame: LocalFrame, timestampMs: Long, options: Map<String, OptionValue>): List<PixelDetection> {
            check(!closed) { "$id used after close" }
            failRun?.let { throw it }
            timestamps += timestampMs
            optionsSeen += options
            return output(frame)
        }

        override fun prepare(action: String) {
            failPrepare?.let { throw it }
        }

        override fun command(
            action: String,
            args: Map<String, OptionValue>,
            frame: LocalFrame?,
            timestampMs: Long,
            options: Map<String, OptionValue>,
        ): CommandOutcome {
            lastCommandFrame = frame
            return CommandOutcome(
                true, "Captured 1 person → 1 files",
                listOf(ServerMessage.CommandResult.ResultFile("photo.jpg", "photo", "file:/cache/captures/20261023-101500-123/photo.jpg")),
            )
        }

        override fun close() {
            closed = true
        }
    }

    private class FakeReader(val options: () -> Map<String, OptionValue>) : TextReader, AutoCloseable {
        var closed = false
        override fun read(still: android.graphics.Bitmap, translateTo: List<String>) = error("not used")
        override fun translate(lines: List<String>, from: List<String?>, to: String) = error("not used")
        override fun languages() = emptyList<TranslationLanguage>()
        override fun close() { closed = true }
    }

    private class FakeLoader : LocalPipelineLoader {
        val readers = mutableListOf<FakeReader>()
        override fun textReader(options: () -> Map<String, OptionValue>): TextReader = FakeReader(options).also { readers += it }

        val created = mutableListOf<FakePipeline>()
        val failures = mutableMapOf<String, Exception>()
        var output: (String, LocalFrame) -> List<PixelDetection> = { _, _ -> emptyList() }

        override fun configKey(pipeline: String, options: Map<String, OptionValue>, gpu: Boolean) =
            "$pipeline/${options[LocalCatalog.OPT_MODEL]}/$gpu"

        override fun load(pipeline: String, options: Map<String, OptionValue>, gpu: Boolean): LocalPipeline {
            failures[pipeline]?.let { throw it }
            return FakePipeline(pipeline) { output(pipeline, it) }.also { created += it }
        }

        fun latest(id: String) = created.last { it.id == id }
    }

    private class Recorder : RecognitionBackend.Listener {
        val states = mutableListOf<ConnectionState>()
        val hellos = mutableListOf<ServerMessage.Hello>()
        val statuses = mutableListOf<ServerMessage.PipelineStatusChange>()
        val detections = mutableListOf<ServerMessage.Detections>()
        val results = mutableListOf<ServerMessage.CommandResult>()
        override fun onStateChanged(state: ConnectionState, detail: String?) { states += state }
        override fun onHello(hello: ServerMessage.Hello) { hellos += hello }
        override fun onPipelineStatus(change: ServerMessage.PipelineStatusChange) { statuses += change }
        override fun onDetections(result: ServerMessage.Detections, receivedAtMs: Long) { detections += result }
        override fun onCommandResult(result: ServerMessage.CommandResult) { results += result }

        fun statusesOf(id: String) = statuses.filter { it.pipeline == id }.map { it.status }
    }

    private class QueueExecutor : Executor {
        val queue = ArrayDeque<Runnable>()
        override fun execute(command: Runnable) { queue += command }
        fun drain() { while (queue.isNotEmpty()) queue.removeFirst().run() }
    }

    private val direct = Executor { it.run() }
    private val recorder = Recorder()
    private val loader = FakeLoader()
    private val stats = StreamStats()
    private var now = 1_000L

    private fun backend(fast: Executor = direct, slow: Executor = direct, load: Executor = direct) =
        LocalBackend(recorder, stats, loader, LocalLanes(fast, slow, load), "0.6.0.3", clock = { now })

    private fun person(x: Float = 100f, conf: Float = 0.9f) = PixelDetection(x, 100f, x + 200f, 400f, conf, "person")

    @Test
    fun start_says_connected_and_hello_then_loads_default_pipelines() {
        val b = backend()
        b.start()
        assertEquals(listOf(ConnectionState.CONNECTED), recorder.states)
        val hello = recorder.hellos.single()
        assertEquals("This phone (CPU)", hello.device)
        assertEquals("0.6.0.3", hello.version)
        assertEquals(LocalCatalog.specs.map { it.id }, hello.pipelines.map { it.id })
        assertTrue(hello.pipelines.all { it.status == PipelineStatus.NOT_LOADED })
        // objects is on by default: loading → ready; the rest stay off.
        assertEquals(listOf(PipelineStatus.LOADING, PipelineStatus.READY), recorder.statusesOf("objects"))
        assertEquals(listOf("objects"), loader.created.map { it.id })
    }

    @Test
    fun a_frame_becomes_a_server_shaped_detections_message() {
        loader.output = { _, _ -> listOf(person()) }
        val b = backend()
        b.start()
        assertTrue(b.offer(Frame(capturedAtMs = 900)))
        val d = recorder.detections.single()
        assertEquals(1L, d.seq)
        assertEquals(900L, d.clientTs)
        assertEquals(640 to 480, d.width to d.height)
        assertEquals(1.5, d.decodeMs, 0.0)
        assertEquals(listOf("objects"), d.pipelineMs.keys.toList())
        assertEquals(NormBox(100f / 640, 100f / 480, 300f / 640, 400f / 480), d.dets.single().box)
        val track = d.tracks.single()
        assertEquals("objects:1", track.trackId)
        assertEquals("objects" to "person", track.pipeline to track.label)
        assertEquals(0, track.seenMsAgo)
        val s = stats.snapshot()
        assertEquals(setOf("objects"), s.pipelineMs.keys)
        assertEquals(1, s.objectCount)
    }

    @Test
    fun global_confidence_gates_new_tracks_only() {
        loader.output = { _, _ -> listOf(person(conf = 0.3f)) }
        val b = backend()
        b.applySettings(StreamSettings(confidence = 0.5f))
        b.start()
        b.offer(Frame())
        assertEquals(1, recorder.detections.last().dets.size) // raw dets keep everything above the floor
        assertTrue(recorder.detections.last().tracks.isEmpty())
    }

    @Test
    fun timestamps_increase_strictly_per_pipeline() {
        val b = backend()
        b.start()
        b.offer(Frame(uptimeMs = 50))
        b.offer(Frame(uptimeMs = 50)) // same camera timestamp
        b.offer(Frame(uptimeMs = 40)) // clock went back
        assertEquals(listOf(50L, 51L, 52L), loader.latest("objects").timestamps)
    }

    @Test
    fun busy_engine_drops_frames_instead_of_queueing() {
        val fast = QueueExecutor()
        val b = backend(fast = fast)
        b.start()
        assertTrue(b.offer(Frame()))
        assertFalse(b.canAcceptFrame())
        assertFalse("the camera's next frame is dropped, not queued", b.offer(Frame()))
        fast.drain()
        assertTrue(b.canAcceptFrame())
        assertTrue(b.offer(Frame()))
        fast.drain()
        assertEquals(2, recorder.detections.size)
    }

    @Test
    fun disabling_unloads_and_enabling_loads_with_options() {
        val b = backend()
        b.start()
        val objects = loader.latest("objects")
        b.setPipelineSettings(PipelineSettings().withEnabled("objects", false).withEnabled("pose", true))
        assertTrue(objects.closed)
        assertEquals(PipelineStatus.NOT_LOADED, recorder.statusesOf("objects").last())
        assertEquals(listOf(PipelineStatus.LOADING, PipelineStatus.READY), recorder.statusesOf("pose"))
        b.offer(Frame())
        assertEquals(setOf("pose"), recorder.detections.last().pipelineMs.keys)
        val opts = loader.latest("pose").optionsSeen.single()
        assertEquals(OptionValue.Bool(true), opts[LocalCatalog.OPT_NAMES])
        assertEquals(OptionValue.Str(LocalCatalog.POSE_LITE), opts[LocalCatalog.OPT_MODEL])
    }

    @Test
    fun a_model_choice_reloads_and_a_plain_option_does_not() {
        val b = backend()
        b.start()
        b.setPipelineSettings(PipelineSettings().withEnabled("pose", true))
        val lite = loader.latest("pose")
        b.setPipelineSettings(PipelineSettings().withEnabled("pose", true).withOption("pose", "names", OptionValue.Bool(false)))
        assertEquals(lite, loader.latest("pose"))
        b.setPipelineSettings(
            PipelineSettings().withEnabled("pose", true).withOption("pose", LocalCatalog.OPT_MODEL, OptionValue.Str(LocalCatalog.POSE_FULL))
        )
        assertTrue(lite.closed)
        assertTrue(loader.latest("pose") !== lite)
        assertEquals(listOf(PipelineStatus.LOADING, PipelineStatus.READY, PipelineStatus.LOADING, PipelineStatus.READY), recorder.statusesOf("pose"))
    }

    @Test
    fun load_failure_reports_a_readable_error_and_retries_after_off_on() {
        loader.failures["faces"] = ModelUnavailableException("No internet — model download failed")
        val b = backend()
        b.start()
        val on = PipelineSettings().withEnabled("faces", true)
        b.setPipelineSettings(on)
        val error = recorder.statuses.last()
        assertEquals(ServerMessage.PipelineStatusChange("faces", PipelineStatus.ERROR, "No internet — model download failed"), error)
        b.setPipelineSettings(on.withEnabled("text", false)) // unrelated change: no retry loop
        assertEquals(1, recorder.statusesOf("faces").count { it == PipelineStatus.LOADING })
        loader.failures.clear()
        b.setPipelineSettings(on.withEnabled("faces", false))
        b.setPipelineSettings(on)
        assertEquals(PipelineStatus.READY, recorder.statusesOf("faces").last())
    }

    @Test
    fun a_pipeline_that_throws_while_running_is_unloaded_with_its_error() {
        val b = backend()
        b.start()
        loader.latest("objects").failRun = IllegalStateException("GPU lost")
        b.offer(Frame())
        assertEquals(ServerMessage.PipelineStatusChange("objects", PipelineStatus.ERROR, "GPU lost"), recorder.statuses.last())
        assertTrue(loader.latest("objects").closed)
        assertTrue(recorder.detections.last().pipelineMs.isEmpty()) // the reply still goes out
    }

    @Test
    fun slow_pipeline_runs_on_its_interval_and_joins_the_replies() {
        loader.output = { id, _ -> if (id == "text") listOf(PixelDetection(10f, 10f, 200f, 60f, 0.95f, "EXIT")) else emptyList() }
        val b = backend()
        b.start()
        b.setPipelineSettings(PipelineSettings().withEnabled("objects", false).withEnabled("text", true))
        now = 1_000
        b.offer(Frame())
        now = 1_100
        b.offer(Frame())
        now = 1_000L + LocalCatalog.TEXT_INTERVAL_MS
        b.offer(Frame())
        assertEquals(2, loader.latest("text").timestamps.size)
        val (first, between, again) = recorder.detections
        assertEquals(setOf("text"), first.pipelineMs.keys)
        assertTrue("no run between intervals", between.pipelineMs.isEmpty())
        assertEquals("EXIT", between.tracks.single().label) // held between runs
        assertEquals(setOf("text"), again.pipelineMs.keys)
        assertEquals("text:1", again.tracks.single().trackId)
    }

    @Test
    fun slow_lane_never_queues_a_second_run() {
        val slow = QueueExecutor()
        val b = backend(slow = slow)
        b.start()
        b.setPipelineSettings(PipelineSettings().withEnabled("text", true))
        b.offer(Frame())
        now += 10_000
        b.offer(Frame()) // interval passed, but the first run hasn't finished
        assertEquals(1, slow.queue.size)
        slow.drain()
        assertEquals(1, loader.latest("text").timestamps.size)
    }

    @Test
    fun actions_run_on_the_newest_frame_and_answer_like_the_server() {
        val b = backend()
        b.start()
        b.setPipelineSettings(PipelineSettings().withEnabled("pose", true))
        val frame = Frame(capturedAtMs = 4242)
        b.offer(frame)
        val req = b.sendCommand("pose", "capture", emptyMap())
        assertNotNull(req)
        val r = recorder.results.single()
        assertEquals(req, r.req)
        assertEquals("pose" to "capture", r.pipeline to r.action)
        assertTrue(r.ok)
        assertEquals("photo.jpg", r.files.single().name)
        assertEquals(frame, loader.latest("pose").lastCommandFrame)
    }

    @Test
    fun actions_on_unavailable_pipelines_fail_politely() {
        val b = backend()
        b.start()
        b.sendCommand("pose", "capture", emptyMap())
        assertEquals("Body pose is off — turn it on first", recorder.results.last().message)
        assertFalse(recorder.results.last().ok)
        b.setPipelineSettings(PipelineSettings().withEnabled("pose", true))
        loader.latest("pose").failPrepare = ModelUnavailableException("No internet — model download failed")
        b.sendCommand("pose", "capture", emptyMap())
        assertEquals("No internet — model download failed", recorder.results.last().message)
        assertNull(b.sendCommand("nope", "x", emptyMap()))
    }

    @Test
    fun stop_refuses_frames_and_commands_and_start_says_hello_again() {
        val b = backend()
        b.start()
        b.stop()
        assertEquals(listOf(ConnectionState.CONNECTED, ConnectionState.DISCONNECTED), recorder.states)
        assertFalse(b.canAcceptFrame())
        assertFalse(b.offer(Frame()))
        assertNull(b.sendCommand("objects", "x", emptyMap()))
        b.start()
        assertEquals(2, recorder.hellos.size)
        assertEquals(PipelineStatus.READY, recorder.hellos.last().pipelines.first { it.id == "objects" }.status)
        assertEquals("models stay loaded across stop/start", 1, loader.created.size)
    }

    @Test
    fun gpu_toggle_reloads_pipelines_and_updates_the_device_line() {
        val b = backend()
        b.start()
        val cpu = loader.latest("objects")
        b.applySettings(StreamSettings(localGpu = true))
        assertEquals("This phone (GPU)", recorder.hellos.last().device)
        assertTrue(cpu.closed)
        assertTrue(loader.latest("objects") !== cpu)
    }

    @Test
    fun shutdown_closes_everything() {
        val b = backend()
        b.start()
        b.setPipelineSettings(PipelineSettings().withEnabled("pose", true))
        b.shutdown()
        assertTrue(loader.created.all { it.closed })
        assertEquals(ConnectionState.DISCONNECTED, recorder.states.last())
    }

    @Test
    fun text_reader_is_created_once_follows_the_text_options_and_closes_on_shutdown() {
        val b = backend()
        b.start()
        assertTrue("created only when the Read button needs it", loader.readers.isEmpty())
        val reader = b.textReader as FakeReader
        assertTrue(b.textReader === reader)
        assertEquals(OptionValue.Str(OcrLanguage.AUTO_CHOICE), reader.options()[LocalCatalog.OPT_OCR_LANGUAGE])
        // The text pipeline doesn't need to be on: the reader uses its OCR language choice either way.
        b.setPipelineSettings(PipelineSettings().withOption(LocalCatalog.TEXT, LocalCatalog.OPT_OCR_LANGUAGE, OptionValue.Str(OcrLanguage.RUSSIAN.choice)))
        assertEquals(OptionValue.Str(OcrLanguage.RUSSIAN.choice), reader.options()[LocalCatalog.OPT_OCR_LANGUAGE])
        b.shutdown()
        assertTrue(reader.closed)
        assertEquals(1, loader.readers.size)
    }

    @Test
    fun a_backend_whose_reader_was_never_used_creates_none() {
        val b = backend()
        b.start()
        b.shutdown()
        assertTrue(loader.readers.isEmpty())
    }
}
