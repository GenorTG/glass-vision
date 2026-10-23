package pl.genor.glassvision

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Shader
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import app.cash.paparazzi.DeviceConfig
import com.android.resources.NightMode

/** Realistic fake data + scaffolding for the Paparazzi snapshots (the camera is a dark placeholder). */
object SnapshotFixtures {
    val PHONE = DeviceConfig.PIXEL_5.copy(nightMode = NightMode.NIGHT)
    const val THEME = "Theme.GlassVision"
    const val VERSION = "0.5.0.1"

    /** Neutral example server (no real hosts or names in snapshot data). */
    const val SERVER_URL = "ws://192.168.1.50:8770/ws"
    val server = StreamSettings(serverUrl = SERVER_URL, token = "example-token")

    /** Pipelines exactly as the live server describes them (captured hello). */
    val hello: ServerMessage.Hello by lazy {
        val text = requireNotNull(javaClass.classLoader?.getResource("hello_v3.json")).readText()
        Protocol.parse(text) as ServerMessage.Hello
    }

    private fun spec(id: String) = hello.pipelines.first { it.id == id }

    /**
     * Mixed live states, in an order that puts one of each state in view: objects ready, faces ready (has
     * options), text loading, pose failed to load, then the rest off/ready.
     */
    val mixedPipelines: List<PipelineSpec> by lazy {
        listOf(
            spec("objects").copy(status = PipelineStatus.READY),
            spec("faces").copy(status = PipelineStatus.READY),
            spec("text").copy(status = PipelineStatus.LOADING),
            spec("pose").copy(status = PipelineStatus.ERROR, statusMessage = "CUDA out of memory"),
            spec("gestures").copy(status = PipelineStatus.READY),
            spec("open_vocab").copy(status = PipelineStatus.NOT_LOADED),
        )
    }

    /** objects (default on) + faces + text + gestures on, pose off (but errored). */
    val mixedSettings = PipelineSettings()
        .withEnabled("faces", true)
        .withEnabled("text", true)
        .withEnabled("gestures", true)

    val liveStats = StatsSnapshot(
        sentFps = 10.0f, resultFps = 9.8f, avgRttMs = 48f, serverMs = 14.2,
        pipelineMs = linkedMapOf("objects" to 8.1, "faces" to 11.3, "gestures" to 5.2, "text" to 55.0),
        dropCount = 0, errorCount = 0, objectCount = 4,
    )

    fun liveState(facing: CameraFacing = CameraFacing.BACK) = MainScreen.State(
        connection = ConnectionState.CONNECTED,
        connectionDetail = SERVER_URL,
        serverInfo = StatusText.serverInfo("0.4.0.3", "cuda:0 (NVIDIA GeForce RTX 3090)"),
        stats = liveStats,
        paused = false,
        pipelines = mixedPipelines,
        pipelineSettings = mixedSettings,
        facing = facing,
        canFlip = true,
    )

    /** A portrait 720×960 frame: a person, their face (known), a cup, and a raised hand with its skeleton. */
    val detections: ServerMessage.Detections by lazy {
        val hand = listOf(
            0.27f to 0.71f,
            0.22f to 0.67f, 0.19f to 0.63f, 0.17f to 0.59f, 0.155f to 0.555f,
            0.24f to 0.60f, 0.235f to 0.555f, 0.232f to 0.525f, 0.23f to 0.495f,
            0.275f to 0.595f, 0.275f to 0.545f, 0.275f to 0.51f, 0.275f to 0.48f,
            0.31f to 0.60f, 0.315f to 0.555f, 0.318f to 0.525f, 0.32f to 0.50f,
            0.34f to 0.615f, 0.35f to 0.58f, 0.355f to 0.56f, 0.36f to 0.54f,
        ).map { (x, y) -> KeyPoint(x, y, 1f) }
        val tracks = listOf(
            Detection("objects", "person", 0.91f, NormBox(0.10f, 0.12f, 0.90f, 0.99f), "objects:7"),
            Detection("faces", "Alex", 0.93f, NormBox(0.38f, 0.20f, 0.64f, 0.42f), "faces:2", detail = "0.71"),
            Detection("objects", "cup", 0.78f, NormBox(0.68f, 0.60f, 0.86f, 0.76f), "objects:9", seenMsAgo = 120),
            Detection("gestures", "Thumb_Up", 0.87f, NormBox(0.14f, 0.47f, 0.38f, 0.73f), "gestures:1", points = hand),
        )
        ServerMessage.Detections(
            seq = 42, clientTs = 0, width = 720, height = 960, decodeMs = 2.1, serverMs = 14.2,
            pipelineMs = mapOf("objects" to 8.1), dets = tracks, tracks = tracks,
        )
    }

    fun catalog(specs: List<PipelineSpec>? = mixedPipelines) = PipelineCatalog().apply {
        if (specs != null) onHello(hello.copy(pipelines = specs))
    }

    class MemoryChoices(var settings: PipelineSettings = mixedSettings) : PipelineChoices {
        override fun load() = settings
        override fun update(transform: (PipelineSettings) -> PipelineSettings) {
            settings = transform(settings)
        }
    }

    class MemoryStream(var settings: StreamSettings = server) : StreamSettingsEditor {
        override fun load() = settings
        override fun update(transform: (StreamSettings) -> StreamSettings) {
            settings = transform(settings).sanitized()
        }
    }

    object NoCallbacks : MainScreen.Callbacks {
        override fun onTogglePipeline(id: String, enabled: Boolean) = Unit
        override fun onTunePipeline(id: String) = Unit
        override fun onAction(entry: ActionEntry) = Unit
        override fun onSettings() = Unit
        override fun onPauseToggle() = Unit
        override fun onFlipCamera() = Unit
        override fun onTakePhoto() = Unit
        override fun onConnectionSetup() = Unit
    }

    /** activity_main without CameraX: placeholder "camera", the real overlay, the real chrome. */
    fun mainScreen(
        context: Context,
        state: MainScreen.State,
        detections: ServerMessage.Detections? = null,
        hudExpanded: Boolean = false,
    ): View {
        val root = FrameLayout(context)
        root.addView(FakeCamera(context), matchParent())
        val overlay = DetectionOverlayView(context)
        overlay.glideMs = 0
        overlay.clock = { 1_000L } // layoutlib's clock jumps past STALE_MS between show() and draw
        overlay.mirrored = state.facing.isMirrored
        state.pipelines?.let(overlay::setPipelines)
        detections?.let(overlay::show)
        root.addView(overlay, matchParent())
        val chrome = LayoutInflater.from(context).inflate(R.layout.view_camera_chrome, root, false)
        root.addView(chrome)
        MainScreen(chrome, NoCallbacks).apply {
            this.hudExpanded = hudExpanded
            render(state)
        }
        return root
    }

    /** A sheet as it appears: camera placeholder, dim, the sheet at the bottom. */
    fun overCamera(context: Context, sheet: View, dim: Int = 0x99): View {
        val root = FrameLayout(context)
        root.addView(FakeCamera(context), matchParent())
        root.addView(View(context).apply { setBackgroundColor(Color.argb(dim, 0, 0, 0)) }, matchParent())
        root.addView(
            sheet,
            FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM),
        )
        return root
    }

    private fun matchParent() = FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)

    /** Dark stand-in for the preview: the 3:4 frame area (fit-center) as a soft gradient, black bars around. */
    private class FakeCamera(context: Context) : View(context) {
        private val paint = Paint(Paint.ANTI_ALIAS_FLAG)

        override fun onDraw(canvas: Canvas) {
            canvas.drawColor(Color.BLACK)
            val r = BoxMapper.fitCenterRect(720, 960, width, height)
            paint.shader = LinearGradient(
                r.left, r.top, r.right, r.bottom,
                Color.rgb(0x2B, 0x33, 0x3B), Color.rgb(0x12, 0x16, 0x1A), Shader.TileMode.CLAMP,
            )
            canvas.drawRect(r.left, r.top, r.right, r.bottom, paint)
        }
    }
}
