package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Test

class GeometryTest {
    private val eps = 1e-3f

    private fun assertRect(expected: ViewRect, actual: ViewRect) {
        assertEquals(expected.left, actual.left, eps)
        assertEquals(expected.top, actual.top, eps)
        assertEquals(expected.right, actual.right, eps)
        assertEquals(expected.bottom, actual.bottom, eps)
    }

    @Test
    fun fit_center_portrait_image_in_tall_view_letterboxes_top_and_bottom() {
        // 720x960 upright image in a 1080x2160 view: scale 1.5 → 1080x1440, centered vertically.
        assertRect(ViewRect(0f, 360f, 1080f, 1800f), BoxMapper.fitCenterRect(720, 960, 1080, 2160))
    }

    @Test
    fun fit_center_landscape_image_in_wide_view_pillarboxes_left_and_right() {
        // 960x720 in 2000x1000: scale min(2.083, 1.389)=1.3889 → 1333.3x1000, x offset 333.3.
        val r = BoxMapper.fitCenterRect(960, 720, 2000, 1000)
        assertRect(ViewRect(333.333f, 0f, 1666.667f, 1000f), r)
    }

    @Test
    fun maps_normalized_box_into_content_rect() {
        val content = BoxMapper.fitCenterRect(720, 960, 1080, 2160)
        val mapped = BoxMapper.mapBox(NormBox(0.12f, 0.30f, 0.48f, 0.97f), content)
        assertRect(ViewRect(129.6f, 360f + 432f, 518.4f, 360f + 1396.8f), mapped)
    }

    @Test
    fun full_box_covers_content_and_out_of_range_is_clamped() {
        val content = ViewRect(10f, 20f, 110f, 220f)
        assertRect(content, BoxMapper.mapBox(NormBox(-0.2f, 0f, 1f, 1.5f), content))
    }

    @Test
    fun mirrored_box_flips_x_edges_and_keeps_y() {
        // Front camera: preview mirrored, frame not → box [x1, x2] lands at [1 - x2, 1 - x1].
        val content = ViewRect(0f, 0f, 100f, 200f)
        val box = NormBox(0.1f, 0.2f, 0.4f, 0.6f)
        assertRect(ViewRect(10f, 40f, 40f, 120f), BoxMapper.mapBox(box, content, mirrored = false))
        assertRect(ViewRect(60f, 40f, 90f, 120f), BoxMapper.mapBox(box, content, mirrored = true))
    }

    @Test
    fun mirrored_mapping_respects_letterbox_offset_and_clamps() {
        val content = BoxMapper.fitCenterRect(720, 960, 1080, 2160) // left 0, top 360, 1080 wide
        // Keypoint at x 0.25 of the frame shows at 75% of the content width when mirrored.
        assertEquals(810f, BoxMapper.mapX(0.25f, content, mirrored = true), eps)
        assertEquals(270f, BoxMapper.mapX(0.25f, content, mirrored = false), eps)
        assertEquals(360f + 0.5f * 1440f, BoxMapper.mapY(0.5f, content), eps)
        // Out-of-range x clamps before mirroring: -0.2 → 0 → right edge.
        assertEquals(1080f, BoxMapper.mapX(-0.2f, content, mirrored = true), eps)
        // A pillarboxed content rect: mirroring stays inside it.
        val wide = ViewRect(100f, 0f, 300f, 100f)
        assertRect(ViewRect(100f, 0f, 300f, 100f), BoxMapper.mapBox(NormBox(0f, 0f, 1f, 1f), wide, mirrored = true))
        assertEquals(280f, BoxMapper.mapX(0.1f, wide, mirrored = true), eps)
    }

    @Test
    fun upright_scaled_size_rotates_and_only_downscales() {
        // Sensor 1280x960 rotated 90° → upright 960x1280 → long side 960 → 720x960.
        assertEquals(FrameSize(720, 960), FrameEncoder.uprightScaledSize(1280, 960, 90, 960))
        assertEquals(FrameSize(960, 720), FrameEncoder.uprightScaledSize(1280, 960, 0, 960))
        assertEquals(FrameSize(480, 640), FrameEncoder.uprightScaledSize(1280, 960, 270, 640))
        // Front sensors usually report 270°: same upright portrait size as the back camera's 90°.
        assertEquals(FrameSize(720, 960), FrameEncoder.uprightScaledSize(1280, 960, 270, 960))
        assertEquals(FrameSize(960, 720), FrameEncoder.uprightScaledSize(1280, 960, 180, 960))
        // Never upscale.
        assertEquals(FrameSize(480, 640), FrameEncoder.uprightScaledSize(640, 480, 90, 1280))
    }

    @Test
    fun settings_sanitize_clamps_ranges() {
        val s = StreamSettings(
            serverUrl = "  ", targetFps = 99, inFlightWindow = 0, sendLongSide = 777,
            jpegQuality = 5, confidence = 2f, modelImgsz = 333, holdMs = 5000,
        ).sanitized()
        assertEquals(1000, s.holdMs)
        assertEquals(0.10f, StreamSettings(confidence = 0.01f).sanitized().confidence, 1e-6f)
        assertEquals(250, StreamSettings(holdMs = 299).sanitized().holdMs)
        assertEquals("", s.serverUrl) // blank = not configured; there is no built-in server
        assertEquals(30, s.targetFps)
        assertEquals(1, s.inFlightWindow)
        assertEquals(960, s.sendLongSide)
        assertEquals(30, s.jpegQuality)
        assertEquals(0.95f, s.confidence, 1e-6f)
        assertEquals(640, s.modelImgsz)
        assertEquals(70, StreamSettings(glideMs = 33).sanitized().glideMs)
        assertEquals(0, StreamSettings(glideMs = 0).sanitized().glideMs)
        assertEquals(CameraFacing.BACK, StreamSettings().cameraFacing)
    }

    @Test
    fun reset_to_defaults_keeps_url_and_camera() {
        val editor = object : StreamSettingsEditor {
            var s = StreamSettings(serverUrl = "ws://x/ws", targetFps = 25, cameraFacing = CameraFacing.FRONT)
            override fun load() = s
            override fun update(transform: (StreamSettings) -> StreamSettings) {
                s = transform(s)
            }
        }
        editor.resetToDefaults()
        assertEquals(StreamSettings(serverUrl = "ws://x/ws", cameraFacing = CameraFacing.FRONT), editor.s)
    }

    @Test
    fun camera_facing_prefs_flip_and_fallback() {
        assertEquals(CameraFacing.FRONT, CameraFacing.fromPref("front"))
        assertEquals(CameraFacing.BACK, CameraFacing.fromPref(null))
        assertEquals(CameraFacing.BACK, CameraFacing.fromPref("garbage"))
        assertEquals(CameraFacing.FRONT, CameraFacing.BACK.flipped())
        assertEquals(CameraFacing.BACK, CameraFacing.FRONT.flipped())
        assertEquals(true, CameraFacing.FRONT.isMirrored)
        assertEquals(false, CameraFacing.BACK.isMirrored)
        val both = setOf(CameraFacing.BACK, CameraFacing.FRONT)
        assertEquals(CameraFacing.FRONT, CameraFacing.resolve(CameraFacing.FRONT, both))
        // A phone without a front camera streams from the back one.
        assertEquals(CameraFacing.BACK, CameraFacing.resolve(CameraFacing.FRONT, setOf(CameraFacing.BACK)))
        assertEquals(null, CameraFacing.resolve(CameraFacing.BACK, emptySet()))
    }
}
