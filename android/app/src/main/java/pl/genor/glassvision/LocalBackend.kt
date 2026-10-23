package pl.genor.glassvision

import androidx.camera.core.ImageProxy
import java.util.concurrent.Executor
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.max

/** One upright camera frame for the on-device engine ([BitmapFrame] on the phone, fakes in JVM tests). */
interface LocalFrame {
    val width: Int
    val height: Int
    /** Wall-clock ms when the camera delivered it → the result's `client_ts` (latency in the HUD). */
    val capturedAtMs: Long
    /** Monotonic ms, for MediaPipe's VIDEO-mode timestamps. */
    val uptimeMs: Long
    /** Converting the camera image (rotate + scale) → the result's `decode_ms`. */
    val prepMs: Double
}

/** Turns a CameraX frame into a [LocalFrame] (long side ≤ [longSide]); analysis thread. */
fun interface LocalFrameSource {
    fun convert(image: ImageProxy, longSide: Int): LocalFrame
}

/** What an action produced: like a server's `command_result` without the request bookkeeping. */
data class CommandOutcome(
    val ok: Boolean,
    val message: String,
    val files: List<ServerMessage.CommandResult.ResultFile> = emptyList(),
)

/** A loaded on-device recognizer. Used from one thread at a time (its lane); closed on that lane. */
interface LocalPipeline : AutoCloseable {
    /** Shown as the pipeline's status message once ready (e.g. "GPU unavailable, using the CPU"). */
    val note: String? get() = null

    /** Results in pixels of [frame]. [timestampMs] strictly increases per instance (MediaPipe VIDEO mode). */
    fun run(frame: LocalFrame, timestampMs: Long, options: Map<String, OptionValue>): List<PixelDetection>

    /** Load thread, before [command]: fetch/create what [action] needs (e.g. a model). Throws with a readable message. */
    fun prepare(action: String) = Unit

    /** Runs [action] on [frame] (the newest one; null before the first). */
    fun command(
        action: String,
        args: Map<String, OptionValue>,
        frame: LocalFrame?,
        timestampMs: Long,
        options: Map<String, OptionValue>,
    ): CommandOutcome = CommandOutcome(false, "Unknown action: $action")
}

/** Creates pipelines; [load] may download a model first. */
interface LocalPipelineLoader {
    /** Identity of the instance [options] + [gpu] need (model choice, creation-time options); a change reloads. */
    fun configKey(pipeline: String, options: Map<String, OptionValue>, gpu: Boolean): String

    /** Blocking (load thread). Throws with a message meant for the user (e.g. "No internet — model download failed"). */
    fun load(pipeline: String, options: Map<String, OptionValue>, gpu: Boolean): LocalPipeline

    /**
     * The Read button's reader, or null if this loader has none; [options] = the text pipeline's current options
     * (its OCR language), read on every call. Closed with the backend if it is [AutoCloseable].
     */
    fun textReader(options: () -> Map<String, OptionValue>): TextReader? = null
}

/** The engine's threads: fast pipelines in sequence, slow ones (OCR) separately, loading/downloads separately. */
class LocalLanes(val fast: Executor, val slow: Executor, val load: Executor, val shutdown: () -> Unit = {})

/**
 * The on-device [RecognitionBackend]: recognition runs on this phone with the pipelines of [LocalCatalog] and
 * talks to the app in the server's message types (hello, pipeline_status, detections, command_result).
 *
 * - Pipelines load lazily on first enable (model download included) on the load lane: status loading → ready, or
 *   error with a readable message; a failed one retries when it is switched off and on again or the app resumes.
 *   Disabling unloads (phones are short on memory). An option that changes the model or a creation-time setting
 *   reloads.
 * - Frames: one in flight; while the fast lane is busy new frames are refused (the camera's latest frame wins, no
 *   queue). Fast pipelines (interval 0) run in sequence on each frame; slow ones get the newest frame on their own
 *   lane at most every `interval_ms` and their results join the next reply, like the server's background lanes.
 * - Each pipeline has a [LocalTracker]; `pipeline_ms` carries real phone timings for the HUD.
 * - Actions run on the pipeline's lane against the newest frame, after [LocalPipeline.prepare] on the load lane.
 */
class LocalBackend(
    private val listener: RecognitionBackend.Listener,
    private val stats: StreamStats,
    private val loader: LocalPipelineLoader,
    private val lanes: LocalLanes,
    private val version: String,
    private val frames: LocalFrameSource? = null,
    private val clock: () -> Long = System::currentTimeMillis,
    private val specs: List<PipelineSpec> = LocalCatalog.specs,
) : RecognitionBackend {
    override val mode: ProcessingMode get() = ProcessingMode.LOCAL

    private class Slot(val spec: PipelineSpec) {
        // Guarded by the backend's lock.
        var status = PipelineStatus.NOT_LOADED
        var message: String? = null
        var instance: LocalPipeline? = null
        var key: String? = null
        var loadingKey: String? = null
        /** Key whose load failed: not retried until something changes. */
        var failedKey: String? = null
        var slowBusy = false
        var lastSlowStartMs = Long.MIN_VALUE / 2
        var measuredMs = 0.0
        var pendingMs: Double? = null
        var pendingDets: List<Detection> = emptyList()

        // Lane-confined.
        var lastTs = 0L
        // Own lock: written by the slow lane, read by the fast lane.
        val tracker = LocalTracker(spec.id)

        val isSlow: Boolean get() = spec.intervalMs > 0
    }

    private val lock = Any()
    private val slots = specs.associateTo(LinkedHashMap()) { it.id to Slot(it) }
    private var started = false
    private var choices = PipelineSettings()
    @Volatile private var conf = StreamSettings().confidence
    @Volatile private var holdMs = StreamSettings().holdMs
    private var gpu = false

    private val fastBusy = AtomicBoolean(false)
    @Volatile private var latestFrame: LocalFrame? = null
    private var seq = 0L
    private val nextReq = AtomicLong(1)

    /** Created on first use (it holds its own models, apart from the live pipeline's), closed in [shutdown]. */
    private val reader = lazy {
        loader.textReader { synchronized(lock) { slots[LocalCatalog.TEXT]?.let(::optionsLocked).orEmpty() } }
    }

    override val textReader: TextReader? get() = reader.value

    // --- lifecycle ---------------------------------------------------------------------------------

    override fun start() {
        val hello = synchronized(lock) {
            if (started) return
            started = true
            slots.values.forEach { it.failedKey = null } // coming back: try failed downloads again
            helloLocked()
        }
        stats.reset()
        listener.onStateChanged(ConnectionState.CONNECTED, DETAIL)
        listener.onHello(hello)
        reconcile()
    }

    override fun stop() {
        synchronized(lock) {
            if (!started) return
            started = false
        }
        listener.onStateChanged(ConnectionState.DISCONNECTED, DETAIL)
    }

    override fun shutdown() {
        stop()
        synchronized(lock) {
            for (slot in slots.values) {
                slot.loadingKey = null
                val old = slot.instance ?: continue
                slot.instance = null
                slot.key = null
                closeOnLane(slot, old)
            }
        }
        if (reader.isInitialized()) (reader.value as? AutoCloseable)?.let(::closeQuietly)
        lanes.shutdown()
    }

    override fun applySettings(settings: StreamSettings) {
        conf = settings.confidence
        holdMs = settings.holdMs
        val hello = synchronized(lock) {
            if (gpu == settings.localGpu) return
            gpu = settings.localGpu
            if (started) helloLocked() else null
        }
        // The device line names the delegate; the pipelines reload with it (their keys include it).
        hello?.let(listener::onHello)
        reconcile()
    }

    override fun setPipelineSettings(settings: PipelineSettings) {
        synchronized(lock) { choices = settings }
        reconcile()
    }

    private fun helloLocked(): ServerMessage.Hello =
        LocalCatalog.hello(version, device(gpu), slots.mapValues { (_, s) -> s.status to s.message })

    // --- loading -----------------------------------------------------------------------------------

    private fun optionsLocked(slot: Slot): Map<String, OptionValue> =
        slot.spec.options.associate { it.key to choices.optionValue(slot.spec, it) }

    /** Brings loaded pipelines in line with the user's choices; loads run on the load lane. */
    private fun reconcile() {
        val events = ArrayList<ServerMessage.PipelineStatusChange>()
        val loads = ArrayList<Runnable>()
        synchronized(lock) {
            if (!started) return
            for (slot in slots.values) {
                if (!choices.isEnabled(slot.spec)) {
                    slot.loadingKey = null
                    slot.failedKey = null
                    val old = slot.instance
                    if (old != null) {
                        slot.instance = null
                        slot.key = null
                        closeOnLane(slot, old)
                        synchronized(slot.tracker) { slot.tracker.clear() }
                    }
                    setStatusLocked(slot, PipelineStatus.NOT_LOADED, null, events)
                    continue
                }
                val options = optionsLocked(slot)
                val key = loader.configKey(slot.spec.id, options, gpu)
                if (key == slot.key && slot.instance != null) continue
                if (key == slot.loadingKey || key == slot.failedKey) continue
                slot.loadingKey = key
                slot.failedKey = null
                setStatusLocked(slot, PipelineStatus.LOADING, null, events)
                val useGpu = gpu
                loads += Runnable { load(slot, key, options, useGpu) }
            }
        }
        events.forEach(listener::onPipelineStatus)
        for (r in loads) {
            try {
                lanes.load.execute(r)
            } catch (_: RejectedExecutionException) {
                return // shut down
            }
        }
    }

    private fun load(slot: Slot, key: String, options: Map<String, OptionValue>, useGpu: Boolean) {
        synchronized(lock) { if (slot.loadingKey != key) return } // superseded while queued
        val events = ArrayList<ServerMessage.PipelineStatusChange>()
        val pipeline = try {
            loader.load(slot.spec.id, options, useGpu)
        } catch (e: Exception) {
            failLoad(slot, key, e, events)
            events.forEach(listener::onPipelineStatus)
            return
        } catch (e: LinkageError) { // native library missing for this CPU
            failLoad(slot, key, e, events)
            events.forEach(listener::onPipelineStatus)
            return
        }
        synchronized(lock) {
            if (slot.loadingKey != key) {
                closeOnLane(slot, pipeline) // switched off or changed meanwhile
                return
            }
            slot.instance?.let { closeOnLane(slot, it) }
            slot.instance = pipeline
            slot.key = key
            slot.loadingKey = null
            synchronized(slot.tracker) { slot.tracker.clear() }
            setStatusLocked(slot, PipelineStatus.READY, pipeline.note, events)
        }
        events.forEach(listener::onPipelineStatus)
    }

    private fun failLoad(slot: Slot, key: String, e: Throwable, events: MutableList<ServerMessage.PipelineStatusChange>) {
        synchronized(lock) {
            if (slot.loadingKey != key) return
            slot.loadingKey = null
            slot.failedKey = key
            setStatusLocked(slot, PipelineStatus.ERROR, readable(e), events)
        }
    }

    /** A pipeline that throws while running is unloaded and shows the error (switch it off and on to retry). */
    private fun failRun(slot: Slot, instance: LocalPipeline, e: Exception) {
        val events = ArrayList<ServerMessage.PipelineStatusChange>()
        synchronized(lock) {
            if (slot.instance !== instance) return
            slot.failedKey = slot.key
            slot.instance = null
            slot.key = null
            closeOnLane(slot, instance)
            setStatusLocked(slot, PipelineStatus.ERROR, readable(e), events)
        }
        events.forEach(listener::onPipelineStatus)
    }

    private fun setStatusLocked(
        slot: Slot,
        status: PipelineStatus,
        message: String?,
        events: MutableList<ServerMessage.PipelineStatusChange>,
    ) {
        if (slot.status == status && slot.message == message) return
        slot.status = status
        slot.message = message
        events += ServerMessage.PipelineStatusChange(slot.spec.id, status, message)
    }

    private fun laneOf(slot: Slot): Executor = if (slot.isSlow) lanes.slow else lanes.fast

    /** Closes after any run already queued on the pipeline's lane (lanes are serial). */
    private fun closeOnLane(slot: Slot, pipeline: LocalPipeline) {
        try {
            laneOf(slot).execute { closeQuietly(pipeline) }
        } catch (_: RejectedExecutionException) {
            closeQuietly(pipeline)
        }
    }

    private fun closeQuietly(closeable: AutoCloseable) {
        try {
            closeable.close()
        } catch (_: Exception) {
            // Nothing useful to do: it is gone either way.
        }
    }

    // --- frames ------------------------------------------------------------------------------------

    override fun canAcceptFrame(): Boolean = synchronized(lock) { started } && !fastBusy.get()

    override fun submitFrame(image: ImageProxy, settings: StreamSettings): Boolean {
        val source = frames ?: return false
        if (!canAcceptFrame() || !fastBusy.compareAndSet(false, true)) return false
        val frame = try {
            source.convert(image, settings.sendLongSide)
        } catch (e: RuntimeException) {
            fastBusy.set(false)
            throw e
        }
        return enqueue(frame)
    }

    /** Runs [frame] if the engine is free (else drops it: the newest frame wins). Any thread. */
    fun offer(frame: LocalFrame): Boolean {
        if (!synchronized(lock) { started } || !fastBusy.compareAndSet(false, true)) return false
        return enqueue(frame)
    }

    /** Called holding [fastBusy]. */
    private fun enqueue(frame: LocalFrame): Boolean {
        latestFrame = frame
        stats.onSent()
        return try {
            lanes.fast.execute {
                try {
                    process(frame)
                } finally {
                    fastBusy.set(false)
                }
            }
            true
        } catch (_: RejectedExecutionException) {
            fastBusy.set(false)
            false
        }
    }

    private class Job(val slot: Slot, val pipeline: LocalPipeline, val options: Map<String, OptionValue>)

    private fun process(frame: LocalFrame) {
        val startNs = System.nanoTime()
        val now = clock()
        val fast = ArrayList<Job>()
        val slow = ArrayList<Job>()
        val confNow = conf
        synchronized(lock) {
            if (!started) return
            for (slot in slots.values) {
                val p = slot.instance ?: continue
                if (!slot.isSlow) {
                    fast += Job(slot, p, optionsLocked(slot))
                } else if (!slot.slowBusy && now - slot.lastSlowStartMs >= slot.spec.intervalMs) {
                    slot.slowBusy = true
                    slot.lastSlowStartMs = now
                    slow += Job(slot, p, optionsLocked(slot))
                }
            }
        }
        for (job in slow) {
            try {
                lanes.slow.execute { runSlow(job, frame) }
            } catch (_: RejectedExecutionException) {
                synchronized(lock) { job.slot.slowBusy = false }
            }
        }

        val ran = LinkedHashMap<String, Double>()
        val dets = ArrayList<Detection>()
        for (job in fast) {
            val ts = nextTimestamp(job.slot, frame)
            val t0 = System.nanoTime()
            val found = try {
                job.pipeline.run(frame, ts, job.options)
            } catch (e: Exception) {
                failRun(job.slot, job.pipeline, e)
                continue
            }
            ran[job.slot.spec.id] = (System.nanoTime() - t0) / 1e6
            val gate = if (job.slot.spec.usesGlobalConf) confNow else null
            synchronized(job.slot.tracker) { job.slot.tracker.update(found, frame.width, frame.height, gate, now) }
            found.mapTo(dets) { it.normalized(job.slot.spec.id, frame.width, frame.height) }
        }

        val tracks = ArrayList<Detection>()
        synchronized(lock) {
            for (slot in slots.values) {
                if (slot.isSlow) {
                    slot.pendingMs?.let { ran[slot.spec.id] = it }
                    slot.pendingMs = null
                    dets += slot.pendingDets
                    slot.pendingDets = emptyList()
                }
            }
        }
        val hold = holdMs
        for (slot in slots.values) {
            val (loaded, measured) = synchronized(lock) { (slot.instance != null) to slot.measuredMs }
            if (!loaded) continue
            val slotHold = LocalTracker.holdFor(hold, slot.spec.intervalMs, measured)
            synchronized(slot.tracker) { tracks += slot.tracker.views(slotHold, now) }
        }

        val engineMs = (System.nanoTime() - startNs) / 1e6
        if (!synchronized(lock) { started }) return // stopped while this frame ran: nobody wants it now
        val result = ServerMessage.Detections(
            seq = ++seq,
            clientTs = frame.capturedAtMs,
            width = frame.width,
            height = frame.height,
            decodeMs = frame.prepMs,
            serverMs = engineMs,
            pipelineMs = ran,
            dets = dets,
            tracks = tracks,
        )
        val received = clock()
        stats.onDetections(received - frame.capturedAtMs, engineMs, ran, tracks.size)
        listener.onDetections(result, received)
    }

    private fun runSlow(job: Job, frame: LocalFrame) {
        val slot = job.slot
        try {
            if (synchronized(lock) { slot.instance !== job.pipeline }) return // unloaded meanwhile
            val ts = nextTimestamp(slot, frame)
            val t0 = System.nanoTime()
            val found = try {
                job.pipeline.run(frame, ts, job.options)
            } catch (e: Exception) {
                failRun(slot, job.pipeline, e)
                return
            }
            val ms = (System.nanoTime() - t0) / 1e6
            val gate = if (slot.spec.usesGlobalConf) conf else null
            synchronized(slot.tracker) { slot.tracker.update(found, frame.width, frame.height, gate, clock()) }
            synchronized(lock) {
                slot.measuredMs = ms
                slot.pendingMs = ms
                slot.pendingDets = found.map { it.normalized(slot.spec.id, frame.width, frame.height) }
            }
        } finally {
            synchronized(lock) { slot.slowBusy = false }
        }
    }

    /** Strictly increasing per pipeline (MediaPipe VIDEO mode rejects repeats), following the frame clock. */
    private fun nextTimestamp(slot: Slot, frame: LocalFrame?): Long {
        val ts = max(frame?.uptimeMs ?: 0L, slot.lastTs + 1)
        slot.lastTs = ts
        return ts
    }

    // --- actions -----------------------------------------------------------------------------------

    override fun sendCommand(pipeline: String, action: String, args: Map<String, OptionValue>): Long? {
        val slot = slots[pipeline] ?: return null
        if (!synchronized(lock) { started }) return null
        val req = nextReq.getAndIncrement()
        fun reply(outcome: CommandOutcome) = listener.onCommandResult(
            ServerMessage.CommandResult(
                req = req, pipeline = pipeline, action = action, ok = outcome.ok, message = outcome.message,
                items = emptyList(), itemAction = null, files = outcome.files,
            )
        )
        try {
            lanes.load.execute {
                val (instance, status) = synchronized(lock) { slot.instance to slot.status }
                if (instance == null) {
                    val why = if (status == PipelineStatus.LOADING) "is still loading" else "is off — turn it on first"
                    return@execute reply(CommandOutcome(false, "${slot.spec.name} $why"))
                }
                try {
                    instance.prepare(action)
                } catch (e: Exception) {
                    return@execute reply(CommandOutcome(false, readable(e)))
                }
                laneOf(slot).execute {
                    val (current, options) = synchronized(lock) { slot.instance to optionsLocked(slot) }
                    if (current !== instance) return@execute reply(CommandOutcome(false, "${slot.spec.name} was turned off"))
                    val frame = latestFrame
                    val outcome = try {
                        instance.command(action, args, frame, nextTimestamp(slot, frame), options)
                    } catch (e: Exception) {
                        CommandOutcome(false, readable(e))
                    }
                    reply(outcome)
                }
            }
        } catch (_: RejectedExecutionException) {
            return null
        }
        return req
    }

    companion object {
        /** Connection detail / HUD line in on-device mode. */
        const val DETAIL = "On this phone"

        /** `hello.device`; the HUD shows the part in parentheses ("v0.6.0.1 · GPU"). */
        fun device(gpu: Boolean): String = "This phone (${if (gpu) "GPU" else "CPU"})"

        fun readable(e: Throwable): String = when (e) {
            is ModelUnavailableException -> e.message.orEmpty()
            is LinkageError -> "Not supported on this phone (${e.message ?: e.javaClass.simpleName})"
            else -> e.message?.takeIf { it.isNotBlank() } ?: e.javaClass.simpleName
        }
    }
}
