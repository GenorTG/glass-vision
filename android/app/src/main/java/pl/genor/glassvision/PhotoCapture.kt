package pl.genor.glassvision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.view.View
import androidx.camera.view.PreviewView
import java.util.Date
import java.util.concurrent.Executor

/**
 * The photo button: what the user sees (camera preview + boxes) as a JPEG in the gallery.
 *
 * [capture] grabs the pixels on the main thread (PreviewView.getBitmap and View.draw must run there; both work in
 * either PreviewView implementation mode), then composites, crops off the letterbox bars, encodes and writes on
 * [io]. [onDone] runs on the main thread.
 */
class PhotoCapture(
    private val saver: MediaSaver,
    private val io: Executor,
) {
    private val main = Handler(Looper.getMainLooper())

    /** [frameSize] = the camera image's size (aspect) as shown under FIT_CENTER; null = portrait 3:4. */
    fun capture(preview: PreviewView, overlay: View, frameSize: Pair<Int, Int>?, onDone: (Result<Uri>) -> Unit) {
        val camera = preview.bitmap ?: return onDone(Result.failure(IllegalStateException("camera not ready")))
        val boxes = if (overlay.width > 0 && overlay.height > 0) {
            Bitmap.createBitmap(overlay.width, overlay.height, Bitmap.Config.ARGB_8888).also { overlay.draw(Canvas(it)) }
        } else {
            null
        }
        val target = CaptureFiles.photo(Date())
        io.execute {
            val result = runCatching {
                val photo = compose(camera, boxes, frameSize)
                try {
                    saver.save(target) { out ->
                        if (!photo.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)) error("JPEG encoding failed")
                    }
                } finally {
                    photo.recycle()
                }
            }
            camera.recycle()
            boxes?.recycle()
            main.post { onDone(result) }
        }
    }

    private fun compose(camera: Bitmap, boxes: Bitmap?, frameSize: Pair<Int, Int>?): Bitmap {
        val (fw, fh) = frameSize ?: (PhotoFrame.DEFAULT_FRAME_W to PhotoFrame.DEFAULT_FRAME_H)
        val crop = PhotoFrame.crop(camera.width, camera.height, fw, fh)
        val out = Bitmap.createBitmap(crop.width, crop.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.BLACK) // transparent (unrendered) preview pixels would encode as black anyway
        canvas.translate(-crop.left.toFloat(), -crop.top.toFloat())
        val full = Rect(0, 0, camera.width, camera.height)
        canvas.drawBitmap(camera, null, full, null)
        // The overlay is laid out over the same area as the preview; scale defensively if sizes ever differ.
        boxes?.let { canvas.drawBitmap(it, null, full, null) }
        return out
    }

    private companion object {
        const val JPEG_QUALITY = 92
    }
}
