package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.util.Calendar
import java.util.TimeZone

class CaptureFilesTest {
    private fun file(name: String, url: String, kind: String = "") = ServerMessage.CommandResult.ResultFile(name, kind, url)

    @Test
    fun plans_images_to_gallery_and_data_to_downloads_per_capture() {
        val id = "20261023-101500-123"
        val targets = CaptureFiles.plan(
            listOf(
                file("photo.jpg", "/captures/$id/photo.jpg", "photo"),
                file("openpose_body.png", "/captures/$id/openpose_body.png", "openpose"),
                file("pose_keypoints.json", "/captures/$id/pose_keypoints.json", "json"),
            ),
            fallbackId = "unused",
        )
        assertEquals(
            listOf(
                SaveTarget("photo.jpg", "image/jpeg", SaveCollection.IMAGES, "GlassVision/$id"),
                SaveTarget("openpose_body.png", "image/png", SaveCollection.IMAGES, "GlassVision/$id"),
                SaveTarget("pose_keypoints.json", "application/json", SaveCollection.DOWNLOADS, "GlassVision/$id"),
            ),
            targets,
        )
        assertEquals("Pictures/GlassVision/$id", targets[0].relativePath)
        assertEquals("Download/GlassVision/$id", targets[2].relativePath)
    }

    @Test
    fun capture_id_from_url_or_fallback() {
        assertEquals("abc", CaptureFiles.captureId("/captures/abc/x.png"))
        assertEquals("abc", CaptureFiles.captureId("https://host/prefix/captures/abc/x.png?sig=1"))
        assertNull(CaptureFiles.captureId("/x.png"))
        assertEquals(
            "GlassVision/20261023-1015",
            CaptureFiles.plan(listOf(file("x.png", "/x.png")), "20261023-1015").single().subfolder,
        )
    }

    @Test
    fun names_cannot_escape_the_folder() {
        assertEquals("passwd", CaptureFiles.safeName("../../etc/passwd"))
        assertEquals("evil.png", CaptureFiles.safeName("..\\..\\evil.png"))
        assertEquals("a_b_c.png", CaptureFiles.safeName("a b:c.png"))
        assertEquals("hidden", CaptureFiles.safeName(".hidden"))
        assertEquals("file", CaptureFiles.safeName(".."))
        assertEquals("GlassVision/x_y", CaptureFiles.plan(listOf(file("a.png", "/captures/x y/a.png")), "f").single().subfolder)
    }

    @Test
    fun mime_types_by_extension() {
        assertEquals("image/jpeg", CaptureFiles.mimeOf("A.JPEG"))
        assertEquals("image/webp", CaptureFiles.mimeOf("a.webp"))
        assertEquals("text/plain", CaptureFiles.mimeOf("notes.txt"))
        assertEquals("application/octet-stream", CaptureFiles.mimeOf("model.bin"))
        assertEquals("application/octet-stream", CaptureFiles.mimeOf("noext"))
    }

    @Test
    fun photo_file_name() {
        val cal = Calendar.getInstance(TimeZone.getDefault()).apply { set(2026, Calendar.OCTOBER, 23, 9, 5, 7) }
        assertEquals(
            SaveTarget("glassvision_20261023_090507.jpg", "image/jpeg", SaveCollection.IMAGES, "GlassVision"),
            CaptureFiles.photo(cal.time),
        )
    }

    @Test
    fun photo_crop_removes_letterbox() {
        // Portrait phone 1080×2340 showing a 3:4 image fit-center: full width, 1440 high, centred.
        assertEquals(PixelRect(0, 450, 1080, 1890), PhotoFrame.crop(1080, 2340, 720, 960))
        // Same aspect: nothing to crop.
        assertEquals(PixelRect(0, 0, 600, 800), PhotoFrame.crop(600, 800, 3, 4))
        // Degenerate input: whole view.
        assertEquals(PixelRect(0, 0, 100, 50), PhotoFrame.crop(100, 50, 0, 0))
    }
}
