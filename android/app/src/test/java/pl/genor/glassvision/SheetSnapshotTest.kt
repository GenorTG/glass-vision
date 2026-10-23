package pl.genor.glassvision

import app.cash.paparazzi.Paparazzi
import org.junit.Rule
import org.junit.Test

/** Bottom sheets as they sit over the camera (the dialog window itself is not rendered by layoutlib). */
class SheetSnapshotTest {
    @get:Rule
    val paparazzi = Paparazzi(deviceConfig = SnapshotFixtures.PHONE, theme = SnapshotFixtures.THEME)

    private val context get() = paparazzi.context
    private val faces get() = SnapshotFixtures.mixedPipelines.first { it.id == "faces" }

    private fun learnFace(facing: CameraFacing) = ActionSheet(
        context, faces, faces.actions.first { it.id == "enroll" },
        initialArgs = mapOf("name" to OptionValue.Str("Alex")),
        facing = facing, frontAvailable = true, onSwitchToFront = {}, onSubmit = {},
    )

    @Test
    fun learn_face_on_back_camera_offers_front() {
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, learnFace(CameraFacing.BACK).sheet.view, dim = 0x26))
    }

    @Test
    fun learn_face_after_capture() {
        val sheet = learnFace(CameraFacing.FRONT)
        sheet.showResult(true, "Learned Alex (2 samples). Add more from other angles for better matching.", snackbar = false)
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, sheet.sheet.view, dim = 0x26))
    }

    @Test
    fun known_faces_list() {
        val sheet = ItemListSheet(context, "faces", "list", emptyMap(), "Known faces") { _, _ -> }
        sheet.bind(
            ServerMessage.CommandResult(
                req = 3, pipeline = "faces", action = "list", ok = true, message = "3 known faces",
                items = listOf(
                    ServerMessage.CommandResult.ResultItem("Alex", "Alex (5 samples)"),
                    ServerMessage.CommandResult.ResultItem("Sam", "Sam (3 samples)"),
                    ServerMessage.CommandResult.ResultItem("Robin", "Robin (1 sample)"),
                ),
                itemAction = ServerMessage.CommandResult.ItemAction("delete", "Forget", confirm = true),
            )
        )
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, sheet.sheet.view))
    }

    @Test
    fun known_faces_empty() {
        val sheet = ItemListSheet(context, "faces", "list", emptyMap(), "Known faces") { _, _ -> }
        sheet.bind(
            ServerMessage.CommandResult(
                req = 3, pipeline = "faces", action = "list", ok = true, message = "No known faces yet",
                items = emptyList(), itemAction = ServerMessage.CommandResult.ItemAction("delete", "Forget", true),
            )
        )
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, sheet.sheet.view))
    }

    /** First run: nothing configured yet. */
    @Test
    fun connect_sheet_first_run() {
        val sheet = ConnectionSheet(context, ServerEndpoint(""), tester = { _, _ -> }, onConnect = {})
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, sheet.sheet.view))
    }

    /** Address typed in short form (helper shows the normalized URL), token set, test passed. */
    @Test
    fun connect_sheet_tested_ok() {
        val sheet = ConnectionSheet(context, ServerEndpoint("192.168.1.50", "example-token"), tester = { _, _ -> }, onConnect = {})
        sheet.form.showResult(
            ConnectionTestResult.Ok(HealthInfo(device = "cuda:0 (NVIDIA GeForce RTX 3090)", version = "0.6.0.1"))
        )
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, sheet.sheet.view))
    }

    /** The server closed the socket with 4401. */
    @Test
    fun connect_sheet_token_rejected() {
        val sheet = ConnectionSheet(context, SnapshotFixtures.server.endpoint, tester = { _, _ -> }, onConnect = {})
        sheet.form.showError(context.getString(R.string.connection_rejected))
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, sheet.sheet.view))
    }

    /** Long-press on a mode chip: that pipeline's card. */
    @Test
    fun pipeline_options_text_ocr() {
        val text = SnapshotFixtures.mixedPipelines.first { it.id == "text" }
        val sheet = PipelineSheet(context, text, SnapshotFixtures.catalog(), SnapshotFixtures.MemoryChoices()) { _, _ -> }
        paparazzi.snapshot(SnapshotFixtures.overCamera(context, sheet.sheet.view))
    }
}
