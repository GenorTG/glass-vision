package pl.genor.glassvision

import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import java.util.Locale

/**
 * Settings as one expanded, scrollable bottom sheet with grouped sections: Processing (server or this phone, GPU,
 * downloaded models), Recognition (every pipeline as a [PipelineCard]), Camera, Detection, Stream, Connection, App.
 * Every control writes to [stream]/[choices] immediately, so the running stream picks changes up live; the server
 * address + token ([ConnectionForm]) only on Connect or when the sheet closes, so the socket doesn't reconnect on
 * every keystroke. On this phone, server-only settings (connection, frames in flight, JPEG quality, model input
 * size) are hidden. Plain-data inputs: snapshot tests render [sheet] as is.
 */
class SettingsSheet(
    private val context: Context,
    private val stream: StreamSettingsEditor,
    choices: PipelineChoices,
    catalog: PipelineCatalog,
    private val availableCameras: Set<CameraFacing>,
    /** Installed version (a parameter, not BuildConfig, so snapshots don't change with every commit). */
    versionName: String,
    onAction: (PipelineSpec, ActionSpec) -> Unit,
    private val onCheckUpdates: () -> Unit,
    /** "Test connection"; null hides the button (snapshots). */
    private val connectionTester: ((ServerEndpoint, (ConnectionTestResult) -> Unit) -> Unit)? = null,
    /** The on-device engine's model files ("Models on this phone"); null hides the list. */
    private val models: ModelShelf? = null,
) {
    val sheet = Sheet(context)
    private val body = LayoutInflater.from(context).inflate(R.layout.sheet_settings_body, null as ViewGroup?, false)
    private val editors = OptionEditors(context)
    private val recognition = RecognitionSection(
        context, body.findViewById(R.id.recognition_cards), catalog, choices, onAction,
    )
    /** The Connection section's form (snapshot tests show a test result in it). */
    var connectionForm: ConnectionForm? = null
        private set

    init {
        sheet.setTitle(context.getString(R.string.settings))
        sheet.setBody(body)
        body.findViewById<MaterialButton>(R.id.recognition_defaults).setOnClickListener {
            choices.reset()
            recognition.rebuild()
        }
        body.findViewById<TextView>(R.id.version_label).text =
            context.getString(R.string.settings_version, versionName)
        body.findViewById<MaterialButton>(R.id.check_updates_button).setOnClickListener { onCheckUpdates() }
        body.findViewById<MaterialButton>(R.id.reset_button).setOnClickListener {
            stream.resetToDefaults()
            bindGroups()
        }
        recognition.rebuild()
        bindGroups()
    }

    fun show() {
        recognition.attach()
        sheet.onDismiss {
            recognition.detach()
            connectionForm?.endpoint()?.let(::applyEndpoint)
        }
        sheet.show(expanded = true)
    }

    private fun bindGroups() {
        val s = stream.load()
        val local = s.processing == ProcessingMode.LOCAL
        bindProcessing(s, body.findViewById(R.id.processing_group))
        bindCamera(s, body.findViewById(R.id.camera_group))
        bindDetection(s, body.findViewById(R.id.detection_group))
        bindStream(s, body.findViewById(R.id.stream_group))
        bindConnection(s, body.findViewById(R.id.connection_group))
        body.findViewById<View>(R.id.section_connection).visibility = if (local) View.GONE else View.VISIBLE
        body.findViewById<TextView>(R.id.version_detail).setText(
            if (local) R.string.settings_version_detail_local else R.string.settings_version_detail
        )
    }

    /** Server / on this phone; on the phone also the GPU switch and the downloaded models. */
    private fun bindProcessing(s: StreamSettings, group: LinearLayout) {
        group.removeAllViews()
        val modes = ProcessingMode.entries
        group += editors.segmented(
            context.getString(R.string.processing_label),
            modes.map { it.prefValue },
            { context.getString(if (it == ProcessingMode.LOCAL.prefValue) R.string.processing_local else R.string.processing_server) },
            s.processing.prefValue,
        ) { v ->
            stream.update { it.copy(processing = ProcessingMode.fromPref(v)) }
            // Rebuild after the click has finished (this listener belongs to a view being removed).
            group.post(::bindGroups)
        }
        val local = s.processing == ProcessingMode.LOCAL
        group += caption(if (local) R.string.processing_local_detail else R.string.processing_server_detail)
        if (!local) return
        group += editors.switch(
            context.getString(R.string.processing_gpu),
            context.getString(R.string.processing_gpu_detail),
            s.localGpu,
        ) { on -> stream.update { it.copy(localGpu = on) } }
        models?.let { bindModels(it, group) }
    }

    private fun bindModels(shelf: ModelShelf, group: LinearLayout) {
        val stored = shelf.downloaded()
        val header = LayoutInflater.from(context).inflate(R.layout.row_model, group, false)
        header.findViewById<TextView>(R.id.row_label).setText(R.string.models_title)
        header.findViewById<TextView>(R.id.row_detail).text = if (stored.isEmpty()) {
            context.getString(R.string.models_empty)
        } else {
            context.getString(R.string.models_total, ModelManager.mb(stored.sumOf { it.bytes }))
        }
        header.findViewById<View>(R.id.row_delete).visibility = View.GONE
        group += header
        for (m in stored) {
            val row = LayoutInflater.from(context).inflate(R.layout.row_model, group, false)
            row.findViewById<TextView>(R.id.row_label).text = m.title
            row.findViewById<TextView>(R.id.row_detail).text = ModelManager.mb(m.bytes)
            row.findViewById<MaterialButton>(R.id.row_delete).apply {
                contentDescription = context.getString(R.string.model_delete, m.title)
                setOnClickListener {
                    shelf.delete(m.id)
                    group.post { bindProcessing(stream.load(), group) }
                }
            }
            group += row
        }
    }

    private fun caption(text: Int): TextView = TextView(context).apply {
        setText(text)
        setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodySmall)
        setTextColor(ContextCompat.getColor(context, R.color.gv_on_surface_variant))
        val pad = (6 * context.resources.displayMetrics.density).toInt()
        setPadding(0, pad, 0, pad)
    }

    private fun bindCamera(s: StreamSettings, group: LinearLayout) {
        group.removeAllViews()
        val facings = CameraFacing.entries.filter { it in availableCameras }.ifEmpty { CameraFacing.entries }
        group += editors.segmented(
            context.getString(R.string.settings_camera),
            facings.map { it.prefValue },
            { context.getString(if (it == CameraFacing.FRONT.prefValue) R.string.settings_camera_front else R.string.settings_camera_back) },
            s.cameraFacing.prefValue,
        ) { v -> stream.update { it.copy(cameraFacing = CameraFacing.fromPref(v)) } }
    }

    private fun bindDetection(s: StreamSettings, group: LinearLayout) {
        group.removeAllViews()
        group += editors.slider(
            context.getString(R.string.settings_confidence),
            NumberSteps(StreamSettings.CONF_MIN.toDouble(), StreamSettings.CONF_MAX.toDouble(), StreamSettings.CONF_STEP.toDouble()),
            s.confidence.toDouble(),
            { String.format(Locale.US, "%.2f", it) },
        ) { v -> stream.update { it.copy(confidence = v.toFloat()) } }
        // On the phone every model has its own fixed input size.
        if (s.processing == ProcessingMode.SERVER) {
            group += editors.dropdown(
                context.getString(R.string.settings_imgsz),
                StreamSettings.IMGSZ_OPTIONS.map(Int::toString),
                { context.getString(R.string.settings_px, it.toInt()) },
                s.modelImgsz.toString(),
            ) { v -> (v as? OptionValue.Str)?.value?.toIntOrNull()?.let { px -> stream.update { it.copy(modelImgsz = px) } } }
        }
        group += editors.switch(
            context.getString(R.string.settings_smoothing),
            context.getString(R.string.settings_smoothing_detail),
            s.smoothing,
        ) { on -> stream.update { it.copy(smoothing = on) } }
        group += intSlider(
            R.string.settings_hold, StreamSettings.HOLD_RANGE, StreamSettings.HOLD_STEP, s.holdMs,
            { context.getString(R.string.settings_ms, it) },
        ) { v -> stream.update { it.copy(holdMs = v) } }
        val glideNames = context.resources.getStringArray(R.array.settings_glide_options)
        group += editors.segmented(
            context.getString(R.string.settings_glide),
            StreamSettings.GLIDE_OPTIONS.map(Int::toString),
            { glideNames.getOrElse(StreamSettings.GLIDE_OPTIONS.indexOf(it.toInt())) { "$it ms" } },
            s.glideMs.toString(),
        ) { v -> stream.update { it.copy(glideMs = v.toInt()) } }
    }

    /** On the phone: no network, so no frames in flight or JPEG quality; the size is what the models get. */
    private fun bindStream(s: StreamSettings, group: LinearLayout) {
        group.removeAllViews()
        val server = s.processing == ProcessingMode.SERVER
        group += intSlider(R.string.settings_target_fps, StreamSettings.FPS_RANGE, 1, s.targetFps, { "$it fps" }) { v ->
            stream.update { it.copy(targetFps = v) }
        }
        if (server) {
            group += intSlider(R.string.settings_window, StreamSettings.WINDOW_RANGE, 1, s.inFlightWindow, Int::toString) { v ->
                stream.update { it.copy(inFlightWindow = v) }
            }
        }
        group += editors.segmented(
            context.getString(if (server) R.string.settings_send_size else R.string.settings_frame_size),
            StreamSettings.SEND_SIZES.map(Int::toString),
            { context.getString(R.string.settings_px, it.toInt()) },
            s.sendLongSide.toString(),
        ) { v -> stream.update { it.copy(sendLongSide = v.toInt()) } }
        if (server) {
            group += intSlider(R.string.settings_jpeg_quality, StreamSettings.QUALITY_RANGE, JPEG_STEP, s.jpegQuality, Int::toString) { v ->
                stream.update { it.copy(jpegQuality = v) }
            }
        }
    }

    private fun bindConnection(s: StreamSettings, group: LinearLayout) {
        group.removeAllViews()
        val form = ConnectionForm(context, s.endpoint, connectionTester, ::applyEndpoint)
        connectionForm = form
        group += form.view
    }

    /** Writes only a real change: the streamer reconnects on every endpoint change. */
    private fun applyEndpoint(endpoint: ServerEndpoint) {
        if (endpoint != stream.load().endpoint) {
            stream.update { it.copy(serverUrl = endpoint.wsUrl, token = endpoint.token) }
        }
    }

    private fun intSlider(
        label: Int,
        range: IntRange,
        step: Int,
        value: Int,
        format: (Int) -> String,
        onChange: (Int) -> Unit,
    ): OptionEditor = editors.slider(
        context.getString(label),
        NumberSteps(range.first.toDouble(), range.last.toDouble(), step.toDouble()),
        value.coerceIn(range).toDouble(),
        { format(it.toInt()) },
    ) { onChange(it.toInt()) }

    private operator fun LinearLayout.plusAssign(editor: OptionEditor) = addView(editor.view)

    private operator fun LinearLayout.plusAssign(view: android.view.View) = addView(view)

    private companion object {
        const val JPEG_STEP = 5
    }
}
