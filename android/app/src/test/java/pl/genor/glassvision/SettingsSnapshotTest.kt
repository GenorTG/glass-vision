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
        R.id.section_recognition, R.id.section_camera, R.id.section_detection,
        R.id.section_stream, R.id.section_connection, R.id.section_app,
    )

    private fun snapshotSections(
        vararg visible: Int,
        pipelines: List<String>? = null,
        connected: Boolean = true,
        prepare: (SettingsSheet) -> Unit = {},
    ) {
        val specs = SnapshotFixtures.mixedPipelines.filter { pipelines == null || it.id in pipelines }
        val settings = SettingsSheet(
            paparazzi.context,
            SnapshotFixtures.MemoryStream(),
            SnapshotFixtures.MemoryChoices(),
            SnapshotFixtures.catalog(if (connected) specs else null),
            availableCameras = CameraFacing.entries.toSet(),
            versionName = SnapshotFixtures.VERSION,
            onAction = { _, _ -> },
            onCheckUpdates = {},
            connectionTester = { _, _ -> },
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

    /** Address + (masked) token, after "Test connection" found a server that rejects the token. */
    @Test
    fun connection_with_token() = snapshotSections(R.id.section_connection) { s ->
        s.connectionForm?.showResult(
            ConnectionTestResult.WrongToken(HealthInfo("cuda:0 (NVIDIA GeForce RTX 3090)", "0.6.0.1"), tokenGiven = true)
        )
    }
}
