package pl.genor.glassvision

import android.view.View
import app.cash.paparazzi.Paparazzi
import com.android.ide.common.rendering.api.SessionParams
import org.junit.Rule
import org.junit.Test

/**
 * The settings sheet, one snapshot per group of sections at content height (Paparazzi 1.3 scales every PNG to
 * ≤ 1000 px, so the whole sheet in one image would be unreadable). Same sheet, other sections hidden.
 */
class SettingsSnapshotTest {
    @get:Rule
    val paparazzi = Paparazzi(
        deviceConfig = SnapshotFixtures.PHONE,
        theme = SnapshotFixtures.THEME,
        renderingMode = SessionParams.RenderingMode.V_SCROLL,
    )

    private val sections = listOf(
        R.id.section_processing, R.id.section_recognition, R.id.section_camera, R.id.section_detection,
        R.id.section_stream, R.id.section_connection, R.id.section_app,
    )

    private fun snapshotSections(
        vararg visible: Int,
        pipelines: List<String>? = null,
        connected: Boolean = true,
        stream: StreamSettings = SnapshotFixtures.server,
        specsOverride: List<PipelineSpec>? = null,
        prepare: (SettingsSheet) -> Unit = {},
    ) {
        val specs = specsOverride ?: SnapshotFixtures.mixedPipelines.filter { pipelines == null || it.id in pipelines }
        val settings = SettingsSheet(
            paparazzi.context,
            SnapshotFixtures.MemoryStream(stream),
            SnapshotFixtures.MemoryChoices(),
            SnapshotFixtures.catalog(if (connected) specs else null),
            availableCameras = CameraFacing.entries.toSet(),
            versionName = SnapshotFixtures.VERSION,
            onAction = { _, _ -> },
            onCheckUpdates = {},
            connectionTester = { _, _ -> },
            models = SnapshotFixtures.ModelsOnPhone,
        )
        prepare(settings)
        val sheet = settings.sheet.view
        for (id in sections) sheet.findViewById<View>(id).visibility = if (id in visible) View.VISIBLE else View.GONE
        paparazzi.snapshot(sheet)
    }

    @Test
    fun recognition_objects_faces_text() =
        snapshotSections(R.id.section_recognition, pipelines = listOf("objects", "faces", "text"))

    @Test
    fun recognition_pose_gestures_find() =
        snapshotSections(R.id.section_recognition, pipelines = listOf("pose", "gestures", "open_vocab"))

    @Test
    fun recognition_not_connected() = snapshotSections(R.id.section_recognition, connected = false)

    @Test
    fun camera_and_detection() = snapshotSections(R.id.section_camera, R.id.section_detection)

    @Test
    fun stream_and_app() = snapshotSections(R.id.section_stream, R.id.section_app)

    /** Processing on the server (the default): mode + what it means. */
    @Test
    fun processing_server() = snapshotSections(R.id.section_processing)

    /** On this phone: GPU switch and "Models on this phone" with sizes and delete buttons. */
    @Test
    fun processing_on_this_phone_with_models() =
        snapshotSections(R.id.section_processing, stream = SnapshotFixtures.onPhone)

    /** On this phone the recognizers come from the phone (faces without naming, a failed model download). */
    @Test
    fun recognition_on_this_phone() = snapshotSections(
        R.id.section_recognition,
        stream = SnapshotFixtures.onPhone,
        specsOverride = SnapshotFixtures.localPipelines.filter { it.id in setOf("pose", "faces", "text") },
    )

    /** Server-only rows (frames in flight, JPEG quality, model input size) are gone on the phone. */
    @Test
    fun on_this_phone_detection_stream_and_app() = snapshotSections(
        R.id.section_detection, R.id.section_stream, R.id.section_app, stream = SnapshotFixtures.onPhone,
    )

    /** Address + (masked) token, after "Test connection" found a server that rejects the token. */
    @Test
    fun connection_with_token() = snapshotSections(R.id.section_connection) { s ->
        s.connectionForm?.showResult(
            ConnectionTestResult.WrongToken(HealthInfo("cuda:0 (NVIDIA GeForce RTX 3090)", "0.6.0.1"), tokenGiven = true)
        )
    }
}
