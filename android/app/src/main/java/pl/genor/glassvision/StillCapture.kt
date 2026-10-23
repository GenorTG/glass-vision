package pl.genor.glassvision

import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.camera.core.ImageProxy
import java.util.concurrent.Executor
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/** A still for Read text; [highRes] = from the still capture (false = an analysis frame, the fallback). */
class Still(val bitmap: Bitmap, val highRes: Boolean)

/**
 * One-shot hand-off of the next analysis frame as an upright full-resolution bitmap: the Read button's fallback when
 * no still capture is bound or it failed. [offer] runs on the analysis thread for every frame (also while paused)
 * and costs one atomic read unless a request is pending.
 */
class AnalysisStill {
    private val pending = AtomicReference<((Result<Bitmap>) -> Unit)?>(null)

    /** [onDone] gets the next frame (analysis thread); a request still pending is replaced. */
    fun request(onDone: (Result<Bitmap>) -> Unit) {
        pending.set(onDone)
    }

    /** Drops [onDone] if it has not been served yet; true if it was still pending. */
    fun cancel(onDone: (Result<Bitmap>) -> Unit): Boolean = pending.compareAndSet(onDone, null)

    /** Analysis thread, before the frame is closed. */
    fun offer(image: ImageProxy) {
        val onDone = pending.getAndSet(null) ?: return
        onDone(runCatching { CameraBinder.upright(image, Int.MAX_VALUE) })
    }
}

/**
 * Gets the Read text still: the camera's high-resolution capture, else (not bound, or it failed) the next analysis
 * frame, else a timeout error. [onDone] runs exactly once, on the main thread.
 */
class StillCapture(
    private val camera: CameraBinder,
    private val analysisStill: AnalysisStill,
    private val worker: Executor,
) {
    private val main = Handler(Looper.getMainLooper())

    fun take(onDone: (Result<Still>) -> Unit) {
        val done = AtomicBoolean(false)
        fun finish(result: Result<Still>) {
            if (done.compareAndSet(false, true)) main.post { onDone(result) }
        }

        fun fallback(reason: Throwable?) {
            val request: (Result<Bitmap>) -> Unit = { r -> finish(r.map { Still(it, highRes = false) }) }
            analysisStill.request(request)
            main.postDelayed({
                if (analysisStill.cancel(request)) {
                    finish(Result.failure(reason ?: TimeoutException("The camera delivered no picture")))
                }
            }, FALLBACK_TIMEOUT_MS)
        }

        val started = camera.takeStill(worker) { result ->
            result.fold(
                onSuccess = { finish(Result.success(Still(it, highRes = true))) },
                onFailure = {
                    Log.w(TAG, "still capture failed, using an analysis frame", it)
                    fallback(it)
                },
            )
        }
        if (!started) fallback(null)
        // A capture that never calls back (HAL hiccup) must not leave the sheet waiting forever.
        main.postDelayed({ finish(Result.failure(TimeoutException("The camera did not take the picture"))) }, TOTAL_TIMEOUT_MS)
    }

    private companion object {
        const val TAG = "StillCapture"
        const val FALLBACK_TIMEOUT_MS = 3000L
        const val TOTAL_TIMEOUT_MS = 10_000L
    }
}
