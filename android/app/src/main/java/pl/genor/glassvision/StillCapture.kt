package pl.genor.glassvision

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapRegionDecoder
import android.graphics.Matrix
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executor
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max
import kotlin.math.roundToInt

/** How a Read still turned out; [sharpness] relative to the best live frame ([BlurCheck]), null = not measured. */
data class StillQuality(val sharpness: Float?, val dark: Boolean, val retaken: Boolean)

/**
 * A still for Read text: the camera's JPEG exactly as captured ([jpeg], stored pixels + [rotationDegrees] clockwise to
 * upright), or an upright analysis frame (the fallback, [highRes] false). Pixels are decoded only when and as large as
 * a reader needs them ([decode], [decodeRegion]); a server gets the JPEG itself with an EXIF orientation
 * ([uploadJpeg]). A 12 MP still is never held as two full-size bitmaps.
 */
class Still private constructor(
    val jpeg: ByteArray?,
    val rotationDegrees: Int,
    private val frame: Bitmap?,
    val highRes: Boolean,
) {
    var quality: StillQuality? = null
        internal set

    private val stored: Pair<Int, Int> by lazy {
        frame?.let { it.width to it.height } ?: run {
            val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeByteArray(jpeg, 0, jpeg!!.size, o)
            o.outWidth to o.outHeight
        }
    }

    /** Upright size in pixels. */
    val width: Int get() = if (rotationDegrees % 180 == 0) stored.first else stored.second
    val height: Int get() = if (rotationDegrees % 180 == 0) stored.second else stored.first

    /** A new upright bitmap (the caller owns it), long side at most [maxLongSide]. */
    fun decode(maxLongSide: Int = Int.MAX_VALUE): Bitmap {
        frame?.let { f ->
            val out = FrameEncoder.uprightBitmap(f, 0, maxLongSide)
            return if (out === f) f.copy(Bitmap.Config.ARGB_8888, false) else out
        }
        val (w, h) = stored
        val o = BitmapFactory.Options().apply { inSampleSize = sampleSize(max(w, h), maxLongSide) }
        val source = BitmapFactory.decodeByteArray(jpeg, 0, jpeg!!.size, o) ?: error("cannot decode the picture")
        return upright(source, rotationDegrees, maxLongSide)
    }

    /**
     * [region] (normalized, upright) as a new upright bitmap at [scale] × full resolution (≤ 1): region-decoded from
     * the JPEG, so a crop of a 12 MP still costs only its own pixels.
     */
    fun decodeRegion(region: NormRect, scale: Float = 1f): Bitmap {
        val storedRegion = region.rotated(-rotationDegrees)
        val (w, h) = stored
        val px = storedRegion.toPixels(w, h)
        val wantLong = max(1, (max(px.width, px.height) * scale).roundToInt())
        frame?.let { f ->
            val crop = Bitmap.createBitmap(f, px.left, px.top, px.width, px.height)
            return upright(crop, 0, wantLong, keepSource = crop === f)
        }
        val decoder = newRegionDecoder(jpeg!!)
        try {
            val o = BitmapFactory.Options().apply { inSampleSize = sampleSize(max(px.width, px.height), wantLong) }
            val source = decoder.decodeRegion(android.graphics.Rect(px.left, px.top, px.right, px.bottom), o)
                ?: error("cannot decode the picture")
            return upright(source, rotationDegrees, wantLong)
        } finally {
            decoder.recycle()
        }
    }

    /**
     * The camera's JPEG with its EXIF orientation set, for a reader that rotates by itself (the server); null for an
     * analysis-frame still. Above [maxBytes] (the server's upload limit) a downscaled re-encode instead.
     */
    fun uploadJpeg(maxBytes: Int = MAX_UPLOAD_BYTES): ByteArray? {
        val bytes = jpeg ?: return null
        if (bytes.size < maxBytes - EXIF_ROOM) return JpegExif.withOrientation(bytes, JpegExif.orientationFor(rotationDegrees))
        var longSide = max(width, height)
        while (true) {
            longSide /= 2
            val small = decode(longSide)
            val out = ByteArrayOutputStream(1 shl 20)
            try {
                small.compress(Bitmap.CompressFormat.JPEG, REENCODE_QUALITY, out)
            } finally {
                small.recycle()
            }
            if (out.size() < maxBytes || longSide < MIN_REENCODE_SIDE) return out.toByteArray()
        }
    }

    companion object {
        /** docs/PROTOCOL.md: `/text/read` refuses bodies over 40 MB. */
        const val MAX_UPLOAD_BYTES = 40 shl 20
        private const val EXIF_ROOM = 64
        private const val REENCODE_QUALITY = 92
        private const val MIN_REENCODE_SIDE = 1024

        fun ofJpeg(jpeg: ByteArray, rotationDegrees: Int) = Still(jpeg, Math.floorMod(rotationDegrees, 360), null, highRes = true)

        /** An already upright analysis frame (the fallback). */
        fun ofFrame(bitmap: Bitmap) = Still(null, 0, bitmap, highRes = false)

        /** Largest power-of-two subsampling that keeps [longSide] at or above [wanted]. */
        fun sampleSize(longSide: Int, wanted: Int): Int {
            var s = 1
            while (longSide / (s * 2) >= wanted) s *= 2
            return s
        }

        /** [source] rotated by [degrees] and scaled to [maxLongSide] in one pass; recycles [source] unless kept. */
        private fun upright(source: Bitmap, degrees: Int, maxLongSide: Int, keepSource: Boolean = false): Bitmap {
            val out = FrameEncoder.uprightBitmap(source, degrees, maxLongSide)
            if (out !== source && !keepSource) source.recycle()
            return if (out === source && keepSource) source.copy(Bitmap.Config.ARGB_8888, false) else out
        }

        @Suppress("DEPRECATION")
        private fun newRegionDecoder(jpeg: ByteArray): BitmapRegionDecoder =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                BitmapRegionDecoder.newInstance(jpeg, 0, jpeg.size)
            } else {
                BitmapRegionDecoder.newInstance(jpeg, 0, jpeg.size, false)
            } ?: error("cannot decode the picture")
    }
}

/** A [TextReader] that takes the camera's JPEG as is and applies its EXIF orientation itself (the server). */
interface JpegTextReader {
    fun readJpeg(jpeg: ByteArray, translateTo: List<String>): TextReading
}

/** Reads [still]: its JPEG for a [JpegTextReader], else an upright bitmap at the read size. */
fun TextReader.readStill(still: Still, translateTo: List<String>): TextReading {
    if (this is JpegTextReader) still.uploadJpeg()?.let { return readJpeg(it, translateTo) }
    val bitmap = still.decode(TextLayout.READ_LONG_SIDE)
    try {
        return read(bitmap, translateTo)
    } finally {
        bitmap.recycle()
    }
}

/**
 * The live frames' sharpness and brightness on the text region while a Read still is being prepared: the reference
 * the still is judged against ([BlurCheck]). [offer] runs on the analysis thread.
 */
class FrameProbe(
    /** Upright analysis-frame region (normalized). */
    val region: NormRect,
) {
    @Volatile var bestSharpness: Float? = null
        private set
    @Volatile var meanLuma: Float? = null
        private set
    /** Long side of the analysis frames measured (the scale the still is compared at). */
    @Volatile var frameLongSide: Int = 0
        private set

    fun offer(image: ImageProxy) {
        val plane = image.planes.firstOrNull() ?: return
        if (plane.pixelStride < 3) return // not RGBA
        // Upright = buffer rotated clockwise by rotationDegrees: undo it to find the region in the buffer.
        val rect = region.rotated(-image.imageInfo.rotationDegrees).toPixels(image.width, image.height)
        val (luma, w, h) = Sharpness.lumaOfRgba(plane.buffer, plane.rowStride, plane.pixelStride, rect)
        val s = Sharpness.of(luma, w, h)
        bestSharpness = max(bestSharpness ?: 0f, s)
        meanLuma = Sharpness.mean(luma)
        frameLongSide = max(image.width, image.height)
    }
}

/**
 * Hand-off of analysis frames to Read text: the next frame as an upright bitmap (the fallback when no still capture
 * is bound or it failed), and the [FrameProbe] that measures frames while a still is prepared. [offer] runs on the
 * analysis thread for every frame (also while paused) and costs two atomic reads when neither is pending.
 */
class AnalysisStill {
    private val pending = AtomicReference<((Result<Bitmap>) -> Unit)?>(null)
    @Volatile var probe: FrameProbe? = null

    /** [onDone] gets the next frame (analysis thread); a request still pending is replaced. */
    fun request(onDone: (Result<Bitmap>) -> Unit) {
        pending.set(onDone)
    }

    /** Drops [onDone] if it has not been served yet; true if it was still pending. */
    fun cancel(onDone: (Result<Bitmap>) -> Unit): Boolean = pending.compareAndSet(onDone, null)

    /** Analysis thread, before the frame is closed. */
    fun offer(image: ImageProxy) {
        probe?.let { p -> runCatching { p.offer(image) }.onFailure { Log.w(TAG, "probe failed", it) } }
        val onDone = pending.getAndSet(null) ?: return
        onDone(runCatching { CameraBinder.upright(image, Int.MAX_VALUE) })
    }

    private companion object {
        const val TAG = "AnalysisStill"
    }
}

/** What the Read UI shows while a still is prepared (main thread). */
interface CaptureFeedback {
    /** Focusing and waiting for the phone to be still. */
    fun onHoldStill() = Unit

    /** The first still was blurry: one more try. */
    fun onRetake() = Unit

    /** The shutter fires now. */
    fun onShutter() = Unit
}

/** The camera screen's side of a Read still (main thread). */
interface CaptureUi : CaptureFeedback {
    /** Show the frame the user aimed at instead of the live preview until [unfreeze]. */
    fun freeze()

    fun unfreeze()

    /** A dark still could use the torch: there is one and it is off. */
    val torchWouldHelp: Boolean
}

/**
 * Takes the Read text still the way a document camera does: focus + expose on the live text marks (or the centre)
 * and wait for the phone to be still, both capped (≤ [FOCUS_TIMEOUT_MS]), then shoot; a still clearly blurrier than
 * the live view of the same text ([BlurCheck]) is retaken once and the sharper one kept. Falls back to the next
 * analysis frame when no still capture is bound or it failed, else a timeout error. [onDone] runs exactly once, on
 * the main thread.
 */
class StillCapture(
    private val camera: CameraBinder,
    private val analysisStill: AnalysisStill,
    private val worker: Executor,
    private val holdStill: HoldStill,
) {
    private val main = Handler(Looper.getMainLooper())

    /** [marks] = live text regions (upright analysis frame, normalized), largest first; empty = the centre. */
    fun take(marks: List<NormRect>, feedback: CaptureFeedback, onDone: (Result<Still>) -> Unit) {
        val done = AtomicBoolean(false)
        val region = TextMarks.union(marks) ?: NormRect.CENTRE
        val probe = FrameProbe(region)
        analysisStill.probe = probe

        fun finish(result: Result<Still>) {
            if (!done.compareAndSet(false, true)) return
            main.post {
                if (analysisStill.probe === probe) analysisStill.probe = null
                camera.cancelFocus()
                onDone(result)
            }
        }

        fun fallback(reason: Throwable?) {
            val request: (Result<Bitmap>) -> Unit = { r -> finish(r.map { Still.ofFrame(it) }) }
            analysisStill.request(request)
            main.postDelayed({
                if (analysisStill.cancel(request)) {
                    finish(Result.failure(reason ?: TimeoutException("The camera delivered no picture")))
                }
            }, FALLBACK_TIMEOUT_MS)
        }

        /** Worker thread: the still's sharpness/brightness on the text region vs the probe's live frames. */
        fun measure(still: Still, shot: CameraBinder.Shot, retaken: Boolean): StillQuality = try {
            val stillRegion = region.rotated(shot.analysisToStillDegrees)
            val longSide = probe.frameLongSide.takeIf { it > 0 } ?: CameraBinder.ANALYSIS_LONG_SIDE
            val scale = (longSide.toFloat() / max(still.width, still.height)).coerceAtMost(1f)
            val crop = still.decodeRegion(stillRegion, scale)
            val pixels = IntArray(crop.width * crop.height)
            crop.getPixels(pixels, 0, crop.width, 0, 0, crop.width, crop.height)
            val luma = Sharpness.lumaOfArgb(pixels)
            val sharpness = Sharpness.of(luma, crop.width, crop.height)
            crop.recycle()
            StillQuality(BlurCheck.ratio(sharpness, probe.bestSharpness), Sharpness.mean(luma) < BlurCheck.DARK_LUMA, retaken)
        } catch (e: Exception) {
            Log.w(TAG, "quality check failed", e)
            StillQuality(null, dark = false, retaken = retaken)
        }

        fun shoot(first: Still?) {
            if (done.get()) return
            feedback.onShutter()
            val started = camera.takeStill(worker) { result ->
                result.fold(
                    onSuccess = { shot ->
                        val still = Still.ofJpeg(shot.jpeg, shot.rotationDegrees)
                        val quality = measure(still, shot, retaken = first != null)
                        still.quality = quality
                        Log.i(TAG, "still ${still.width}x${still.height} rot ${shot.rotationDegrees} target ${shot.targetRotationDegrees}: $quality (live ${probe.bestSharpness})")
                        when {
                            first == null && quality.sharpness != null && quality.sharpness < BlurCheck.BLURRY_RATIO ->
                                main.post {
                                    if (done.get()) return@post
                                    feedback.onRetake()
                                    holdStill.await { shoot(still) }
                                }
                            first != null && (first.quality?.sharpness ?: 0f) > (quality.sharpness ?: 0f) ->
                                finish(Result.success(first))
                            else -> finish(Result.success(still))
                        }
                    },
                    onFailure = {
                        if (first != null) {
                            finish(Result.success(first))
                        } else {
                            Log.w(TAG, "still capture failed, using an analysis frame", it)
                            fallback(it)
                        }
                    },
                )
            }
            if (!started) {
                if (first != null) finish(Result.success(first)) else fallback(null)
            }
        }

        feedback.onHoldStill()
        // Focus/exposure on the text and the steady hand run side by side; shoot when both are done.
        val started = SystemClock.uptimeMillis()
        var waiting = 2
        fun ready(what: String) {
            Log.i(TAG, "$what after ${SystemClock.uptimeMillis() - started} ms")
            if (--waiting == 0) shoot(null)
        }
        Log.i(TAG, "preparing: ${marks.size} text marks, region $region")
        camera.focusOn(marks.ifEmpty { listOf(NormRect.CENTRE) }, FOCUS_TIMEOUT_MS) { ready("focused=$it") }
        holdStill.await { ready("still=$it") }
        // A capture that never calls back (HAL hiccup) must not leave the Read flow waiting forever.
        main.postDelayed({ finish(Result.failure(TimeoutException("The camera did not take the picture"))) }, TOTAL_TIMEOUT_MS)
    }

    private companion object {
        const val TAG = "StillCapture"
        const val FOCUS_TIMEOUT_MS = 800L
        const val FALLBACK_TIMEOUT_MS = 3000L
        const val TOTAL_TIMEOUT_MS = 10_000L
    }
}
