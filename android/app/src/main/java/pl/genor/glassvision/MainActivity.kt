package pl.genor.glassvision

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.android.material.snackbar.Snackbar
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Wires CameraX ([CameraBinder]) → [FrameAnalyzer] → the active [RecognitionBackend] ([FrameStreamer] to a server,
 * or [LocalBackend] on this phone, swapped by [BackendSwitch] when Settings → Processing changes) → overlay, and
 * the stores/backend state → [MainScreen] (chips, actions, status) and the sheets (connection, settings,
 * per-pipeline, actions); the photo button → [PhotoCapture], result files → [ResultFileSaver]. Owns no UI logic
 * of its own beyond routing.
 */
class MainActivity :
    AppCompatActivity(),
    RecognitionBackend.Listener,
    MainScreen.Callbacks,
    CameraBinder.Listener,
    CommandUi.CameraHost {

    private lateinit var root: View
    private lateinit var preview: PreviewView
    private lateinit var overlay: DetectionOverlayView
    private lateinit var flash: View
    private lateinit var screen: MainScreen
    private lateinit var camera: CameraBinder

    private lateinit var store: SettingsStore
    private val stats = StreamStats()
    private lateinit var backends: BackendSwitch
    private lateinit var models: ModelManager
    private lateinit var captureStore: CaptureStore
    private lateinit var analysisExecutor: ExecutorService
    private lateinit var mediaExecutor: ExecutorService
    private lateinit var updates: UpdateController
    private lateinit var pipelineStore: PipelineStore
    private val catalog = PipelineCatalog()
    private lateinit var commands: CommandUi
    private lateinit var photos: PhotoCapture
    private lateinit var resultFiles: ResultFileSaver
    private val connectionTester = ConnectionTester()
    private var connectionSheet: ConnectionSheet? = null

    @Volatile private var settings = StreamSettings()
    @Volatile private var paused = false
    private var connection = ConnectionState.DISCONNECTED
    private var connectionDetail: String? = null
    private var serverInfo: String? = null

    private var cameraPermitted = false
    /** Last facing asked of [camera]; [boundFacing] is what it actually bound (may fall back). */
    private var requestedFacing: CameraFacing? = null
    private var boundFacing: CameraFacing? = null
    private var cameras: Set<CameraFacing> = emptySet()

    /** Runs once WRITE_EXTERNAL_STORAGE is granted (API 26–28 only). */
    private var afterStoragePermission: (() -> Unit)? = null

    private val onSettingsChanged: (StreamSettings) -> Unit = ::applySettings
    private val onPipelinesChanged: (PipelineSettings) -> Unit = {
        backends.current?.setPipelineSettings(it)
        render()
    }
    private val onCatalogChanged: (PipelineCatalog.Change) -> Unit = { render() }

    private val hudTicker = object : Runnable {
        override fun run() {
            render()
            root.postDelayed(this, HUD_INTERVAL_MS)
        }
    }

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else showMessage(getString(R.string.camera_permission_denied), true)
        }

    private val requestStoragePermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            val next = afterStoragePermission
            afterStoragePermission = null
            if (granted) next?.invoke() else showMessage(getString(R.string.storage_permission_denied), true)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        root = findViewById(R.id.main_root)
        preview = findViewById(R.id.preview_view)
        overlay = findViewById(R.id.overlay_view)
        flash = findViewById(R.id.capture_flash)
        screen = MainScreen(findViewById(R.id.chrome_root), this)

        store = SettingsStore(this)
        pipelineStore = PipelineStore(this)
        models = ModelManager(
            File(filesDir, MODELS_DIR),
            OkHttpDownloader(),
            LocalModel.entries.map { it.spec } + OcrLanguage.entries.map { it.spec },
        )
        captureStore = CaptureStore(File(cacheDir, CAPTURES_DIR))
        // Backend events arrive on the main thread (the switch hops and drops a replaced backend's late events).
        backends = BackendSwitch(this, ContextCompat.getMainExecutor(this), ::createBackend)
        analysisExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "gv-analysis") }
        mediaExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "gv-media") }
        val saver = MediaSaver(this)
        photos = PhotoCapture(saver, mediaExecutor)
        resultFiles = ResultFileSaver(saver, mediaExecutor)
        updates = UpdateController(this) { settings.endpoint }
        commands = CommandUi(
            this, catalog, { p, a, args -> backends.current?.sendCommand(p, a, args) }, ::showMessage, ArgMemory(this), this,
        )
        camera = CameraBinder(
            this, this, preview, analysisExecutor,
            newAnalyzer = { FrameAnalyzer({ backends.current }, { settings }, { paused }) },
            listener = this,
        )

        applySettings(store.load())
        store.addListener(onSettingsChanged)
        pipelineStore.addListener(onPipelinesChanged)
        catalog.addListener(onCatalogChanged)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
        // First run: ask where the server is, or run on this phone (the camera permission dialog shows on top first).
        if (savedInstanceState == null && settings.processing == ProcessingMode.SERVER && !settings.endpoint.isConfigured) {
            root.post { openConnectionSheet(null) }
        }
    }

    override fun onStart() {
        super.onStart()
        backends.start()
        root.post(hudTicker)
        updateKeepScreenOn()
    }

    override fun onResume() {
        super.onResume()
        updates.onResume()
    }

    override fun onStop() {
        root.removeCallbacks(hudTicker)
        backends.stop()
        overlay.clear()
        super.onStop()
    }

    override fun onDestroy() {
        store.removeListener(onSettingsChanged)
        pipelineStore.removeListener(onPipelinesChanged)
        catalog.removeListener(onCatalogChanged)
        backends.shutdown()
        analysisExecutor.shutdown()
        mediaExecutor.shutdown() // lets a running save finish
        connectionTester.shutdown()
        updates.shutdown()
        super.onDestroy()
    }

    private fun createBackend(mode: ProcessingMode, listener: RecognitionBackend.Listener): RecognitionBackend =
        when (mode) {
            ProcessingMode.SERVER -> FrameStreamer(listener, stats)
            ProcessingMode.LOCAL -> {
                val fast = Executors.newSingleThreadExecutor { r -> Thread(r, "gv-local-fast") }
                val slow = Executors.newSingleThreadExecutor { r -> Thread(r, "gv-local-slow") }
                val load = Executors.newSingleThreadExecutor { r -> Thread(r, "gv-local-load") }
                // shutdown() lets queued work (closing the models) finish.
                val lanes = LocalLanes(fast, slow, load) { listOf(fast, slow, load).forEach(ExecutorService::shutdown) }
                LocalBackend(
                    listener, stats, OnDeviceLoader(this, models, captureStore), lanes, BuildConfig.VERSION_NAME, CameraFrames,
                )
            }
        }

    /**
     * Live-apply: a processing-mode change swaps the backend; otherwise each backend takes what concerns it (only an
     * endpoint change reconnects, only a GPU change reloads models); only a camera change rebinds.
     */
    private fun applySettings(new: StreamSettings) {
        settings = new
        overlay.smoothing = new.smoothing
        overlay.glideMs = new.glideMs
        val switched = backends.select(new.processing) {
            it.applySettings(new)
            it.setPipelineSettings(pipelineStore.load())
        }
        if (switched) {
            // New engine, new pipelines: nothing from the old one may stay on screen.
            overlay.clear()
            catalog.clear()
            stats.reset()
            serverInfo = null
            connection = ConnectionState.DISCONNECTED
            connectionDetail = null
            connectionSheet?.sheet?.dismiss()
            render()
        } else {
            backends.current?.applySettings(new)
        }
        if (cameraPermitted && new.cameraFacing != requestedFacing) bindCamera(new.cameraFacing)
    }

    private fun startCamera() {
        cameraPermitted = true
        bindCamera(settings.cameraFacing)
    }

    private fun bindCamera(facing: CameraFacing) {
        requestedFacing = facing
        camera.bind(facing)
    }

    // CameraBinder.Listener

    override fun onCameraBound(facing: CameraFacing, available: Set<CameraFacing>) {
        boundFacing = facing
        cameras = available
        // New scene: old boxes are in the other camera's coordinates (and mirroring).
        overlay.clear()
        overlay.mirrored = facing.isMirrored
        commands.onCameraChanged(facing)
        render()
    }

    override fun onCameraError(message: String) = showMessage(message, true)

    // CommandUi.CameraHost

    override val facing: CameraFacing get() = boundFacing ?: settings.cameraFacing
    override val availableCameras: Set<CameraFacing> get() = cameras

    override fun switchCamera(facing: CameraFacing) = store.update { it.copy(cameraFacing = facing) }

    // MainScreen.Callbacks

    override fun onTogglePipeline(id: String, enabled: Boolean) = pipelineStore.update { it.withEnabled(id, enabled) }

    override fun onTunePipeline(id: String) {
        val spec = catalog.find(id) ?: return
        PipelineSheet(this, spec, catalog, pipelineStore, commands::run).show()
    }

    override fun onAction(entry: ActionEntry) = commands.run(entry.pipeline, entry.action)

    override fun onSettings() {
        SettingsSheet(
            this, store, pipelineStore, catalog, cameras, BuildConfig.VERSION_NAME, commands::run, ::checkForUpdates,
            connectionTester::test, models,
        ).show()
    }

    /** Updates come from a glass-vision server; on this phone without one there is nothing to ask. */
    private fun checkForUpdates() {
        if (settings.endpoint.isConfigured) updates.checkNow() else showMessage(getString(R.string.update_no_server), false)
    }

    override fun onPauseToggle() {
        paused = !paused
        updateKeepScreenOn()
        render()
    }

    override fun onFlipCamera() = switchCamera(facing.flipped())

    override fun onTakePhoto() = withStorageAccess {
        photos.capture(preview, overlay, overlay.frameSize) { result ->
            if (isFinishing) return@capture
            result.fold(
                onSuccess = { uri -> showMediaMessage(getString(R.string.photo_saved), false, uri, "image/jpeg") },
                onFailure = { showMessage(getString(R.string.photo_failed, it.message ?: it.javaClass.simpleName), true) },
            )
        }
        playFlash()
    }

    override fun onConnectionSetup() = openConnectionSheet(
        if (connection == ConnectionState.UNAUTHORIZED) getString(R.string.connection_rejected) else null,
    )

    private fun openConnectionSheet(error: String?) {
        connectionSheet?.takeIf { it.sheet.isShowing }?.let { open ->
            error?.let(open.form::showError)
            return
        }
        // No server yet, or it can't be reached: offer to run on this phone instead (not for a rejected token).
        val offerLocal = !settings.endpoint.isConfigured || connection == ConnectionState.DISCONNECTED
        val sheet = ConnectionSheet(
            this, settings.endpoint, connectionTester::test,
            onRunLocal = if (offerLocal) ({ store.update { it.copy(processing = ProcessingMode.LOCAL) } }) else null,
        ) { endpoint ->
            store.update { it.copy(serverUrl = endpoint.wsUrl, token = endpoint.token) }
        }
        error?.let(sheet.form::showError)
        sheet.sheet.onDismiss { if (connectionSheet === sheet) connectionSheet = null }
        connectionSheet = sheet
        sheet.sheet.show()
    }

    private fun playFlash() {
        flash.animate().cancel()
        flash.alpha = FLASH_ALPHA
        flash.visibility = View.VISIBLE
        flash.animate().alpha(0f).setDuration(FLASH_MS).withEndAction { flash.visibility = View.GONE }.start()
    }

    /** Gallery writes need WRITE_EXTERNAL_STORAGE only before Android 10 (scoped MediaStore after). */
    private fun withStorageAccess(block: () -> Unit) {
        val granted = Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
        if (granted) {
            block()
        } else {
            afterStoragePermission = block
            requestStoragePermission.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }
    }

    private fun updateKeepScreenOn() {
        if (paused) {
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        } else {
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }

    // RecognitionBackend.Listener — on the main thread, only for the current backend (BackendSwitch).

    override fun onStateChanged(state: ConnectionState, detail: String?) {
        val wasUnauthorized = connection == ConnectionState.UNAUTHORIZED
        connection = state
        connectionDetail = detail
        if (state != ConnectionState.CONNECTED) {
            overlay.clear()
            catalog.clear()
        }
        // Up and running: look for a newer APK (rate-limited inside) on the server, if there is one.
        if (state == ConnectionState.CONNECTED && settings.endpoint.isConfigured) updates.checkQuietly()
        // Token rejected: ask for it where it is entered.
        if (state == ConnectionState.UNAUTHORIZED && !wasUnauthorized && !isFinishing) {
            openConnectionSheet(getString(R.string.connection_rejected))
        }
        render()
    }

    override fun onHello(hello: ServerMessage.Hello) {
        serverInfo = StatusText.serverInfo(hello.version, hello.device)
        overlay.setPipelines(hello.pipelines)
        catalog.onHello(hello)
    }

    override fun onPipelineStatus(change: ServerMessage.PipelineStatusChange) {
        catalog.onStatus(change)
        // Loading errors (out of VRAM, no internet for a model download…) matter even with no sheet open.
        if (change.status == PipelineStatus.ERROR) {
            val name = catalog.find(change.pipeline)?.name ?: change.pipeline
            val msg = change.message ?: getString(R.string.recognition_status_error_plain)
            showMessage(getString(R.string.pipeline_error_message, name, msg), true)
        }
    }

    override fun onDetections(result: ServerMessage.Detections, receivedAtMs: Long) = overlay.show(result)

    override fun onCommandResult(result: ServerMessage.CommandResult) {
        if (isFinishing) return
        commands.onResult(result)
        if (result.ok && result.files.isNotEmpty()) saveResultFiles(result.files)
    }

    /** Any action's `files` (e.g. a pose capture) land in the gallery / Downloads; no pipeline-specific code. */
    private fun saveResultFiles(files: List<ServerMessage.CommandResult.ResultFile>) {
        val endpoint = settings.endpoint
        // On-device results are files in our capture folder; only then may file: URLs be read.
        val localRoot = captureStore.root.takeIf { backends.current?.mode == ProcessingMode.LOCAL }
        withStorageAccess {
            resultFiles.save(endpoint, files, localRoot) { outcome ->
                if (isFinishing) return@save
                val total = outcome.saved + outcome.failed
                when {
                    outcome.failed == 0 -> showMediaMessage(
                        resources.getQuantityString(R.plurals.files_saved, outcome.saved, outcome.saved),
                        false, outcome.firstImage, "image/*",
                    )
                    outcome.saved == 0 -> showMessage(getString(R.string.files_failed, outcome.error.orEmpty()), true)
                    else -> showMediaMessage(
                        getString(R.string.files_partly_saved, outcome.saved, total, outcome.error.orEmpty()),
                        true, outcome.firstImage, "image/*",
                    )
                }
            }
        }
    }

    private fun render() {
        if (isFinishing) return
        screen.render(
            MainScreen.State(
                connection = connection,
                connectionDetail = connectionDetail,
                serverInfo = serverInfo,
                stats = stats.snapshot(),
                paused = paused,
                pipelines = catalog.specs,
                pipelineSettings = pipelineStore.load(),
                facing = facing,
                canFlip = cameras.size > 1,
                mode = settings.processing,
            )
        )
    }

    private fun showMessage(text: String, error: Boolean) = showMediaMessage(text, error, null, null)

    /** Snackbar above the bottom controls; with [openUri] an "Open" action shows it in the gallery/viewer. */
    private fun showMediaMessage(text: String, error: Boolean, openUri: Uri?, mimeType: String?) {
        val bar = Snackbar.make(root, text, if (error || openUri != null) Snackbar.LENGTH_LONG else Snackbar.LENGTH_SHORT)
        bar.anchorView = findViewById(R.id.bottom_bar)
        if (openUri != null) bar.setAction(R.string.open_media) { openMedia(openUri, mimeType) }
        bar.show()
    }

    private fun openMedia(uri: Uri, mimeType: String?) {
        val intent = Intent(Intent.ACTION_VIEW).setDataAndType(uri, mimeType).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        try {
            startActivity(intent)
        } catch (_: ActivityNotFoundException) {
            showMessage(getString(R.string.photo_saved), false)
        }
    }

    private companion object {
        const val HUD_INTERVAL_MS = 250L
        const val FLASH_ALPHA = 0.85f
        const val FLASH_MS = 220L
        /** On-device models (app-specific storage, kept until deleted in Settings). */
        const val MODELS_DIR = "models"
        /** On-device pose captures before they are copied to the gallery (cache: the system may clear it). */
        const val CAPTURES_DIR = "captures"
    }
}
