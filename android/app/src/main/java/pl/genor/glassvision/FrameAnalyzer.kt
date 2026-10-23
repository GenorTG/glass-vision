package pl.genor.glassvision

import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

/**
 * CameraX analyzer (runs on the single analysis executor): gates frames on pause, the FPS cap and the active
 * backend's capacity (in-flight window / engine busy) *before* any per-frame work, then hands the frame to that
 * backend on this same thread (it encodes + sends, or converts + runs on the phone).
 */
class FrameAnalyzer(
    private val backend: () -> RecognitionBackend?,
    private val settings: () -> StreamSettings,
    private val isPaused: () -> Boolean,
    /** Read text's fallback still: served from the next frame, paused or not. */
    private val still: AnalysisStill? = null,
) : ImageAnalysis.Analyzer {
    private var nextDueUptime = 0L

    override fun analyze(image: ImageProxy) {
        try {
            still?.offer(image)
            maybeSend(image)
        } catch (e: RuntimeException) {
            Log.e(TAG, "frame encode/send failed", e)
        } finally {
            // Closing releases the buffer so KEEP_ONLY_LATEST can hand us the next frame.
            image.close()
        }
    }

    private fun maybeSend(image: ImageProxy) {
        if (isPaused()) return
        val s = settings()
        val now = SystemClock.uptimeMillis()
        if (now < nextDueUptime) return
        val target = backend() ?: return
        if (!target.canAcceptFrame()) return

        if (target.submitFrame(image, s)) nextDueUptime = nextDue(nextDueUptime, now, 1000L / s.targetFps)
    }

    companion object {
        /**
         * Fixed-rate schedule: advance from the previous due time, not from `now`, so camera frame jitter
         * doesn't round every gap up (30 fps camera + "last send + 100 ms" gives 7.5 fps, not 10).
         * After a stall (paused, window full) restart from `now` instead of bursting to catch up.
         */
        fun nextDue(previousDue: Long, now: Long, intervalMs: Long): Long =
            if (now - previousDue > intervalMs) now + intervalMs else previousDue + intervalMs
    }

}

private const val TAG = "FrameAnalyzer"
