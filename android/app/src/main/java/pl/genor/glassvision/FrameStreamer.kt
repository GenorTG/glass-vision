package pl.genor.glassvision

import android.util.Log
import androidx.camera.core.ImageProxy
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/**
 * [NOT_CONFIGURED] = no server address yet (nothing connects); [UNAUTHORIZED] = the server rejected the access
 * token (no retries until the endpoint changes).
 */
enum class ConnectionState { NOT_CONFIGURED, DISCONNECTED, CONNECTING, CONNECTED, UNAUTHORIZED }

/**
 * One WebSocket to the glass-vision server (docs/PROTOCOL.md): sends header+JPEG frames, parses replies,
 * auto-reconnects with 1s→5s backoff and enforces the in-flight window. The access token rides in the URL
 * query ([ServerEndpoint.socketUrl]); an auth rejection ([ConnectionLoss]) stops the retry loop.
 *
 * Every frame gets exactly one reply (detections/dropped/error), so `outstanding.size` is the number
 * of frames the server still owes us. New frames are refused while it is >= the window, which keeps
 * latency bounded instead of letting a queue build up.
 *
 * Listener callbacks arrive on OkHttp/scheduler threads, never the main thread. The [RecognitionBackend] for
 * [ProcessingMode.SERVER].
 */
class FrameStreamer(
    private val listener: RecognitionBackend.Listener,
    private val stats: StreamStats,
    private val clock: () -> Long = System::currentTimeMillis,
) : RecognitionBackend {
    override val mode: ProcessingMode get() = ProcessingMode.SERVER

    /** JPEG encoder for [submitFrame]; only touched on the analysis thread. */
    private val encoder = FrameEncoder()

    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .pingInterval(10, TimeUnit.SECONDS)
        .build()
    private val scheduler = Executors.newSingleThreadScheduledExecutor { r ->
        Thread(r, "gv-reconnect").apply { isDaemon = true }
    }

    private val lock = Any()
    // Guarded by [lock].
    private var socket: WebSocket? = null
    private var generation = 0
    private var running = false
    private var endpoint = ServerEndpoint("")
    private var backoffMs = BACKOFF_MIN_MS
    private var reconnectFuture: ScheduledFuture<*>? = null

    @Volatile var state: ConnectionState = ConnectionState.DISCONNECTED
        private set
    @Volatile private var window = 2
    @Volatile private var conf = 0.35f
    @Volatile private var imgsz = 640
    @Volatile private var holdMs = 300

    // Pipeline config, guarded by [lock]. [specs] = pipelines from this connection's hello (null before it);
    // [sentPipelines] = what the server has been told, so later changes go out as deltas.
    private var pipelineSettings = PipelineSettings()
    private var specs: List<PipelineSpec>? = null
    private var sentPipelines: Map<String, PipelineConfig>? = null

    /** seq → send time (ms) for frames that have not been answered yet. */
    private val outstanding = ConcurrentHashMap<Long, Long>()
    private val nextSeq = AtomicLong(0)
    private val nextReq = AtomicLong(1)

    /** Set from a hello that declares `text_read`; cleared whenever the connection (or its endpoint) goes away. */
    @Volatile private var reader: ServerTextReader? = null

    override val textReader: TextReader? get() = reader

    fun start(server: ServerEndpoint) {
        synchronized(lock) {
            endpoint = server
            if (running) return
            running = true
            backoffMs = BACKOFF_MIN_MS
            connectLocked()
        }
    }

    /** Connects to the endpoint last given to [setEndpoint]/[applySettings]. */
    override fun start() = start(synchronized(lock) { endpoint })

    override fun stop() {
        synchronized(lock) {
            running = false
            reconnectFuture?.cancel(false)
            reconnectFuture = null
            generation++
            reader = null
            socket?.close(1000, "client stop")
            socket = null
        }
        setState(ConnectionState.DISCONNECTED, null)
    }

    override fun shutdown() {
        stop()
        scheduler.shutdownNow()
        client.dispatcher.executorService.shutdown()
    }

    /** Reconnects only when the URL or token actually changed. */
    fun setEndpoint(server: ServerEndpoint) {
        synchronized(lock) {
            if (server == endpoint) return
            endpoint = server
            if (!running) return
            reconnectFuture?.cancel(false)
            generation++
            reader = null
            socket?.cancel()
            socket = null
            backoffMs = BACKOFF_MIN_MS
            connectLocked()
        }
    }

    override fun applySettings(settings: StreamSettings) {
        setWindow(settings.inFlightWindow)
        setDetectorConfig(settings.confidence, settings.modelImgsz, settings.holdMs)
        setEndpoint(settings.endpoint)
    }

    fun setWindow(size: Int) {
        window = size.coerceIn(StreamSettings.WINDOW_RANGE)
    }

    /** Stores the detector config and pushes it to the server now if connected (else on next connect). */
    fun setDetectorConfig(confidence: Float, modelImgsz: Int, trackHoldMs: Int) {
        if (confidence == conf && modelImgsz == imgsz && trackHoldMs == holdMs) return
        conf = confidence
        imgsz = modelImgsz
        holdMs = trackHoldMs
        val ws = synchronized(lock) { socket.takeIf { state == ConnectionState.CONNECTED } }
        ws?.send(Protocol.configJson(conf, imgsz, holdMs))
    }

    /** Stores the user's pipeline choices; once the server's hello is known, sends what changed. */
    override fun setPipelineSettings(settings: PipelineSettings) {
        synchronized(lock) {
            pipelineSettings = settings
            if (state == ConnectionState.CONNECTED) socket?.let(::sendPipelineConfigLocked)
        }
    }

    /**
     * Sends a pipeline action (runs on the newest frame the server got). Returns the request id echoed in
     * the `command_result`, or null when not connected.
     */
    override fun sendCommand(pipeline: String, action: String, args: Map<String, OptionValue>): Long? {
        val ws = synchronized(lock) { socket.takeIf { state == ConnectionState.CONNECTED } } ?: return null
        val req = nextReq.getAndIncrement()
        return if (ws.send(Protocol.commandJson(req, pipeline, action, args))) req else null
    }

    /** The text pipeline's current choices, for Read text (the server reads the still with the same script/model). */
    private fun readOptions(): ReadOptions = synchronized(lock) {
        val text = specs?.firstOrNull { it.id == TEXT_PIPELINE } ?: return ReadOptions()
        fun value(key: String) = text.options.firstOrNull { it.key == key }
            ?.let { (pipelineSettings.optionValue(text, it) as? OptionValue.Str)?.value }
        ReadOptions(languages = value("languages"), model = value("translation_model"), translateFrom = value("translate_from"))
    }

    private fun sendPipelineConfigLocked(ws: WebSocket) {
        val known = specs ?: return
        val full = pipelineSettings.resolve(known)
        val delta = PipelineSettings.delta(sentPipelines, full)
        if (delta.isEmpty()) return
        if (ws.send(Protocol.configJson(pipelines = delta))) sentPipelines = full
    }

    override fun canAcceptFrame(): Boolean = canSend()

    /** Encodes the frame (upright, scaled, JPEG) and sends it. */
    override fun submitFrame(image: ImageProxy, settings: StreamSettings): Boolean =
        sendFrame(encoder.encode(image, settings.sendLongSide, settings.jpegQuality))

    /** True if connected and the in-flight window has room. Cheap; call before encoding. */
    fun canSend(): Boolean {
        if (state != ConnectionState.CONNECTED) return false
        expireStalled(clock())
        return outstanding.size < window
    }

    fun sendFrame(frame: EncodedFrame): Boolean {
        val ws = synchronized(lock) { socket.takeIf { state == ConnectionState.CONNECTED } } ?: return false
        if (ws.queueSize() > MAX_SOCKET_QUEUE_BYTES) return false
        val now = clock()
        val seq = nextSeq.getAndIncrement() and 0xFFFF_FFFFL
        val payload = Protocol.encodeFrame(seq, now, frame.bytes, frame.length)
        outstanding[seq] = now
        if (!ws.send(payload.toByteString())) {
            outstanding.remove(seq)
            return false
        }
        stats.onSent()
        return true
    }

    /** Safety valve: if the server never answers a frame, don't let it pin a window slot forever. */
    private fun expireStalled(now: Long) {
        val it = outstanding.entries.iterator()
        while (it.hasNext()) {
            val e = it.next()
            if (now - e.value > STALL_TIMEOUT_MS) {
                Log.w(TAG, "frame seq=${e.key} unanswered for ${now - e.value} ms, releasing slot")
                it.remove()
            }
        }
    }

    private fun connectLocked() {
        val gen = ++generation
        if (!endpoint.isConfigured) {
            socket = null
            setState(ConnectionState.NOT_CONFIGURED, null)
            return
        }
        val request = try {
            Request.Builder().url(endpoint.socketUrl()).build()
        } catch (e: IllegalArgumentException) {
            socket = null
            setState(ConnectionState.DISCONNECTED, "bad URL")
            return
        }
        // Detail = the URL without the token (it ends up in the HUD).
        setState(ConnectionState.CONNECTING, endpoint.wsUrl)
        socket = client.newWebSocket(request, SocketListener(gen))
    }

    private fun scheduleReconnectLocked(reason: String) {
        if (!running) return
        val delay = backoffMs
        backoffMs = (backoffMs * 2).coerceAtMost(BACKOFF_MAX_MS)
        setState(ConnectionState.DISCONNECTED, "$reason · retry ${delay / 1000.0}s")
        reconnectFuture = scheduler.schedule({
            synchronized(lock) { if (running && socket == null) connectLocked() }
        }, delay, TimeUnit.MILLISECONDS)
    }

    private fun setState(newState: ConnectionState, detail: String?) {
        state = newState
        listener.onStateChanged(newState, detail)
    }

    private fun handleMessage(text: String) {
        when (val msg = Protocol.parse(text)) {
            is ServerMessage.Hello -> {
                synchronized(lock) {
                    specs = msg.pipelines
                    sentPipelines = null
                    reader = if (msg.canReadText) ServerTextReader(endpoint, ::readOptions) else null
                    socket?.let(::sendPipelineConfigLocked)
                }
                listener.onHello(msg)
            }
            is ServerMessage.PipelineStatusChange -> listener.onPipelineStatus(msg)
            is ServerMessage.Detections -> {
                outstanding.remove(msg.seq)
                val now = clock()
                stats.onDetections(now - msg.clientTs, msg.serverMs, msg.pipelineMs, msg.tracks.size)
                listener.onDetections(msg, now)
            }
            is ServerMessage.CommandResult -> listener.onCommandResult(msg)
            is ServerMessage.Dropped -> {
                outstanding.remove(msg.seq)
                stats.onDropped()
            }
            is ServerMessage.Error -> {
                msg.seq?.let(outstanding::remove)
                stats.onError()
                Log.w(TAG, "server error seq=${msg.seq}: ${msg.message}")
            }
            null -> Log.w(TAG, "unparseable message: ${text.take(200)}")
        }
    }

    private inner class SocketListener(private val gen: Int) : WebSocketListener() {
        private fun isCurrent() = gen == generation

        override fun onOpen(webSocket: WebSocket, response: Response) {
            synchronized(lock) {
                if (!isCurrent()) return
                backoffMs = BACKOFF_MIN_MS
                outstanding.clear()
                nextSeq.set(0)
                stats.reset()
                // Pipelines follow once hello says which exist (unknown ids would be an `error`).
                specs = null
                sentPipelines = null
                reader = null
                webSocket.send(Protocol.configJson(conf, imgsz, holdMs))
                setState(ConnectionState.CONNECTED, endpoint.wsUrl)
            }
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            if (synchronized(lock) { isCurrent() }) handleMessage(text)
        }

        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
            webSocket.close(1000, null)
            // Decide on the server's close code here: onClosed may report our own reply instead.
            onLost("closed $code", ConnectionLoss.reaction(code, null))
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) =
            onLost("closed $code", ConnectionLoss.reaction(code, null))

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            val reason = response?.let { "HTTP ${it.code}" } ?: t.message ?: t.javaClass.simpleName
            onLost(reason, ConnectionLoss.reaction(null, response?.code))
        }

        private fun onLost(reason: String, reaction: ConnectionLoss.Reaction) {
            synchronized(lock) {
                if (!isCurrent()) return
                generation++ // later callbacks of this socket (onClosed after onClosing) are stale
                socket = null
                reader = null
                outstanding.clear()
                when (reaction) {
                    ConnectionLoss.Reaction.RECONNECT -> scheduleReconnectLocked(reason)
                    // Retrying with the same token can't succeed; wait for the user to change it.
                    ConnectionLoss.Reaction.AUTH_FAILED -> setState(ConnectionState.UNAUTHORIZED, reason)
                }
            }
        }
    }

    private companion object {
        const val TAG = "FrameStreamer"
        const val TEXT_PIPELINE = "text"
        const val BACKOFF_MIN_MS = 1000L
        const val BACKOFF_MAX_MS = 5000L
        const val STALL_TIMEOUT_MS = 3000L
        const val MAX_SOCKET_QUEUE_BYTES = 2L * 1024 * 1024
    }
}
