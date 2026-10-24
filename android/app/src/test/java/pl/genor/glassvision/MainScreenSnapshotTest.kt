package pl.genor.glassvision

import app.cash.paparazzi.Paparazzi
import org.junit.Rule
import org.junit.Test

/** The camera screen (PNG goldens in src/test/snapshots/images; record: ./gradlew recordPaparazziDebug). */
class MainScreenSnapshotTest {
    @get:Rule
    val paparazzi = Paparazzi(deviceConfig = SnapshotFixtures.PHONE, theme = SnapshotFixtures.THEME)

    private val context get() = paparazzi.context

    @Test
    fun live_back_camera() {
        paparazzi.snapshot(SnapshotFixtures.mainScreen(context, SnapshotFixtures.liveState(), SnapshotFixtures.detections))
    }

    /** Same result on the front camera: positions mirrored like the selfie preview, labels still readable. */
    @Test
    fun live_front_camera_mirrored() {
        paparazzi.snapshot(
            SnapshotFixtures.mainScreen(context, SnapshotFixtures.liveState(CameraFacing.FRONT), SnapshotFixtures.detections)
        )
    }

    /** Read text with the torch on next to it (the camera has a flash). */
    @Test
    fun read_text_with_torch_on() {
        val state = SnapshotFixtures.liveState().copy(canReadText = true, canTorch = true, torchOn = true)
        paparazzi.snapshot(SnapshotFixtures.mainScreen(context, state, SnapshotFixtures.detections))
    }

    /** Recognition on the phone: "On device" pill, the phone's pipelines, a named pose, a held OCR line. */
    @Test
    fun live_on_this_phone() {
        paparazzi.snapshot(SnapshotFixtures.mainScreen(context, SnapshotFixtures.localState(), SnapshotFixtures.localDetections))
    }

    /** The HUD on the phone: engine (delegate), latency, engine time and real per-pipeline phone timings. */
    @Test
    fun on_this_phone_expanded_hud() {
        paparazzi.snapshot(
            SnapshotFixtures.mainScreen(context, SnapshotFixtures.localState(), SnapshotFixtures.localDetections, hudExpanded = true)
        )
    }

    @Test
    fun status_expanded_hud() {
        paparazzi.snapshot(
            SnapshotFixtures.mainScreen(context, SnapshotFixtures.liveState(), SnapshotFixtures.detections, hudExpanded = true)
        )
    }

    @Test
    fun paused_only_objects_no_actions() {
        val state = SnapshotFixtures.liveState().copy(paused = true, pipelineSettings = PipelineSettings())
        paparazzi.snapshot(SnapshotFixtures.mainScreen(context, state))
    }

    @Test
    fun offline() {
        val state = SnapshotFixtures.liveState().copy(
            connection = ConnectionState.DISCONNECTED,
            connectionDetail = "failed to connect to /192.168.1.50 · retry 4.0s",
            serverInfo = null,
            stats = StatsSnapshot(0f, 0f, null, null, emptyMap(), 0, 0, 0),
            pipelines = null,
        )
        paparazzi.snapshot(SnapshotFixtures.mainScreen(context, state))
    }

    /** First run: no server yet. The pill and the banner both open the connection sheet. */
    @Test
    fun not_configured() {
        val state = SnapshotFixtures.liveState().copy(
            connection = ConnectionState.NOT_CONFIGURED,
            connectionDetail = null,
            serverInfo = null,
            stats = StatsSnapshot(0f, 0f, null, null, emptyMap(), 0, 0, 0),
            pipelines = null,
        )
        paparazzi.snapshot(SnapshotFixtures.mainScreen(context, state))
    }

    @Test
    fun connecting_single_camera() {
        val state = SnapshotFixtures.liveState().copy(
            connection = ConnectionState.CONNECTING,
            serverInfo = null,
            stats = StatsSnapshot(0f, 0f, null, null, emptyMap(), 0, 0, 0),
            pipelines = null,
            canFlip = false,
        )
        paparazzi.snapshot(SnapshotFixtures.mainScreen(context, state))
    }
}
