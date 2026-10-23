package pl.genor.glassvision

import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

/**
 * CameraX analyzer (runs on the single analysis executor): gates frames on pause, the FPS cap and the
 * streamer's in-flight window *before* encoding, then encodes + sends on this same thread.
 */
class FrameAnalyzer(
    private val streamer: FrameStreamer,
    private val settings: () -> StreamSettings,
    private val isPaused: () -> Boolean,
) : ImageAnalysis.Analyzer {
    private val encoder = FrameEncoder()
    private var nextDueUptime = 0L

    override fun analyze(image: ImageProxy) {
        try {
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
        if (!streamer.canSend()) return

        val frame = encoder.encode(image, s.sendLongSide, s.jpegQuality)
        if (streamer.sendFrame(frame)) nextDueUptime = nextDue(nextDueUptime, now, 1000L / s.targetFps)
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
