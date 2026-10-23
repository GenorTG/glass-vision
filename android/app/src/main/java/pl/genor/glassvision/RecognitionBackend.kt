package pl.genor.glassvision

import androidx.camera.core.ImageProxy
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicInteger

/** Where recognition runs: your glass-vision server, or this phone. */
enum class ProcessingMode(val prefValue: String) {
    SERVER("server"),
    LOCAL("local");

    companion object {
        fun fromPref(value: String?): ProcessingMode = entries.firstOrNull { it.prefValue == value } ?: SERVER
    }
}

/**
 * What recognizes the camera frames: the WebSocket to a server ([FrameStreamer]) or the on-device engine
 * ([LocalBackend]). Both speak the server's message types (docs/PROTOCOL.md: hello with pipeline specs,
 * pipeline_status, detections with tracks/dets/pipeline_ms, command_result with files), so everything above this
 * interface — chips, overlay, settings sheets, actions, HUD, photo button, file saving — is the same in both modes.
 */
interface RecognitionBackend {
    interface Listener {
        fun onStateChanged(state: ConnectionState, detail: String?)
        fun onHello(hello: ServerMessage.Hello)
        fun onPipelineStatus(change: ServerMessage.PipelineStatusChange)
        fun onDetections(result: ServerMessage.Detections, receivedAtMs: Long)
        fun onCommandResult(result: ServerMessage.CommandResult)
    }

    val mode: ProcessingMode

    /** Begin (or resume) delivering results; idempotent. */
    fun start()

    /** Pause: no frames, no results (models may stay loaded). Emits DISCONNECTED. */
    fun stop()

    /** Release everything; the instance is not used again. */
    fun shutdown()

    /** Takes what applies to this backend (endpoint, window, confidence, hold, GPU…); cheap if nothing changed. */
    fun applySettings(settings: StreamSettings)

    /** The user's pipeline choices (enabled + options). */
    fun setPipelineSettings(settings: PipelineSettings)

    /** Runs a pipeline action on the newest frame; the request id echoed in `command_result`, or null if not ready. */
    fun sendCommand(pipeline: String, action: String, args: Map<String, OptionValue>): Long?

    /** Analysis thread, before any per-frame work: is there room for another frame right now? */
    fun canAcceptFrame(): Boolean

    /** Analysis thread: takes the frame (encode+send or convert+run). True if it was accepted. */
    fun submitFrame(image: ImageProxy, settings: StreamSettings): Boolean

    /** The Read button's backend (docs/PROTOCOL.md "Read text"), or null while unsupported (e.g. an older server). */
    val textReader: TextReader? get() = null
}

/**
 * Owns the active [RecognitionBackend] and swaps it when the processing mode changes: the old one is shut down
 * before the new one starts, and only the current one's events reach [listener] — delivered on [deliver]
 * (the main thread in the app) and checked again there, so a result the old backend posted just before the switch
 * can't land after it. Call [select]/[start]/[stop]/[shutdown] on the main thread.
 */
class BackendSwitch(
    private val listener: RecognitionBackend.Listener,
    private val deliver: Executor,
    private val create: (ProcessingMode, RecognitionBackend.Listener) -> RecognitionBackend,
) {
    @Volatile var current: RecognitionBackend? = null
        private set
    private val generation = AtomicInteger()
    private var running = false

    /**
     * Makes [mode]'s backend current; [configure] runs on the new one before it starts (settings, pipeline choices).
     * Returns false (and does nothing) if [mode] is already current.
     */
    fun select(mode: ProcessingMode, configure: (RecognitionBackend) -> Unit = {}): Boolean {
        if (current?.mode == mode) return false
        val gen = generation.incrementAndGet()
        current?.shutdown()
        val next = create(mode, Gated(gen))
        configure(next)
        current = next
        if (running) next.start()
        return true
    }

    fun start() {
        running = true
        current?.start()
    }

    fun stop() {
        running = false
        current?.stop()
    }

    fun shutdown() {
        running = false
        generation.incrementAndGet()
        current?.shutdown()
        current = null
    }

    /** Forwards [gen]'s events while it is still the current generation, checked before and after the hop. */
    private inner class Gated(private val gen: Int) : RecognitionBackend.Listener {
        private fun live() = gen == generation.get()

        private fun post(block: RecognitionBackend.Listener.() -> Unit) {
            if (!live()) return
            deliver.execute { if (live()) listener.block() }
        }

        override fun onStateChanged(state: ConnectionState, detail: String?) = post { onStateChanged(state, detail) }
        override fun onHello(hello: ServerMessage.Hello) = post { onHello(hello) }
        override fun onPipelineStatus(change: ServerMessage.PipelineStatusChange) = post { onPipelineStatus(change) }
        override fun onDetections(result: ServerMessage.Detections, receivedAtMs: Long) =
            post { onDetections(result, receivedAtMs) }
        override fun onCommandResult(result: ServerMessage.CommandResult) = post { onCommandResult(result) }
    }
}
