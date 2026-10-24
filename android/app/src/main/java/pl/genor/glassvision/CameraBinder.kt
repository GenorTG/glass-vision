package pl.genor.glassvision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.util.Log
import android.util.Size
import android.view.Surface
import androidx.camera.core.Camera
import androidx.camera.core.CameraInfoUnavailableException
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.SurfaceOrientedMeteringPointFactory
import androidx.camera.core.TorchState
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executor
import java.util.concurrent.TimeUnit

/**
 * CameraX binding for preview + analysis (+ a high-resolution still for Read text, taken at [captureRotation], with
 * focus/metering on the text and the torch) on the back or front camera.
 * [bind] (re)binds everything on the chosen camera with fresh use cases and a fresh analyzer (so the frame schedule
 * restarts), falling back to the other camera if the device lacks it, and to preview + analysis alone if the device
 * can't add the still capture to that combination. Main thread only.
 */
class CameraBinder(
    private val context: Context,
    private val owner: LifecycleOwner,
    private val previewView: PreviewView,
    private val analysisExecutor: Executor,
    private val newAnalyzer: () -> ImageAnalysis.Analyzer,
    private val listener: Listener,
) {
    interface Listener {
        /** Bound to [facing]; [available] = the cameras this device has. */
        fun onCameraBound(facing: CameraFacing, available: Set<CameraFacing>)

        fun onCameraError(message: String)

        /** The torch went on/off (also off by itself on a rebind). */
        fun onTorchChanged(on: Boolean) = Unit
    }

    /**
     * A still as the camera delivered it: [jpeg] stored pixels, [rotationDegrees] clockwise to upright at the target
     * rotation the shot was taken with ([targetRotationDegrees]; CameraX semantics).
     */
    class Shot(
        val jpeg: ByteArray,
        val rotationDegrees: Int,
        val targetRotationDegrees: Int,
        val facing: CameraFacing,
    ) {
        /**
         * Clockwise rotation from the live (portrait) analysis frame to this still, both upright: the still turned
         * with the phone. CameraX: back camera rotation = sensor − target, front = sensor + target.
         */
        val analysisToStillDegrees: Int
            get() = Math.floorMod(if (facing == CameraFacing.FRONT) targetRotationDegrees else -targetRotationDegrees, 360)
    }

    private var provider: ProcessCameraProvider? = null
    private var requested: CameraFacing? = null
    private var still: ImageCapture? = null
    private var analysis: ImageAnalysis? = null
    private var camera: Camera? = null
    private var boundFacing: CameraFacing? = null

    /** Size of the bound still capture (sensor orientation), or null when only preview + analysis are bound. */
    val stillResolution: Size? get() = still?.resolutionInfo?.resolution

    /**
     * Surface.ROTATION_* the still is taken at ([CaptureOrientation]: how the phone is held); preview and analysis stay
     * portrait. Kept across rebinds.
     */
    var captureRotation: Int = Surface.ROTATION_0
        set(value) {
            field = value
            still?.targetRotation = value
        }

    /** The bound camera has a flash unit (torch available). */
    val hasFlash: Boolean get() = camera?.cameraInfo?.hasFlashUnit() == true

    val torchOn: Boolean get() = camera?.cameraInfo?.torchState?.value == TorchState.ON

    fun setTorch(on: Boolean) {
        val c = camera ?: return
        if (c.cameraInfo.hasFlashUnit()) c.cameraControl.enableTorch(on)
    }

    /**
     * AF + AE on [regions] (upright analysis frame, normalized; the first counts most, the device uses as many as it
     * supports), then [onDone] on the main thread once focused, failed or after [timeoutMs]. Best effort: a fixed-focus
     * camera or no camera calls back at once. Main thread.
     */
    fun focusOn(regions: List<NormRect>, timeoutMs: Long, onDone: (focused: Boolean) -> Unit) {
        val main = ContextCompat.getMainExecutor(context)
        val c = camera
        val a = analysis
        val rotation = a?.resolutionInfo?.rotationDegrees
        if (c == null || a == null || rotation == null || regions.isEmpty()) {
            main.execute { onDone(false) }
            return
        }
        // Points in the analysis buffer (normalized), which is the upright frame turned back by its rotation.
        val factory = SurfaceOrientedMeteringPointFactory(1f, 1f, a)
        val points = regions.map { r ->
            val b = r.rotated(-rotation)
            val size = maxOf(b.right - b.left, b.bottom - b.top).coerceIn(MIN_METERING_SIZE, MAX_METERING_SIZE)
            factory.createPoint((b.left + b.right) / 2f, (b.top + b.bottom) / 2f, size)
        }
        val flags = FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE
        val action = FocusMeteringAction.Builder(points.first(), flags).apply {
            points.drop(1).forEach { addPoint(it, flags) }
            setAutoCancelDuration(FOCUS_HOLD_S, TimeUnit.SECONDS)
        }.build()
        if (!c.cameraInfo.isFocusMeteringSupported(action)) {
            main.execute { onDone(false) }
            return
        }
        var called = false
        fun finish(focused: Boolean) {
            if (called) return
            called = true
            onDone(focused)
        }
        val future = c.cameraControl.startFocusAndMetering(action)
        future.addListener({ finish(runCatching { future.get().isFocusSuccessful }.getOrDefault(false)) }, main)
        previewView.postDelayed({ finish(false) }, timeoutMs)
    }

    /** Back to continuous focus/auto exposure after a [focusOn]. */
    fun cancelFocus() {
        camera?.cameraControl?.cancelFocusAndMetering()
    }

    /**
     * Takes one high-resolution still at [captureRotation] and delivers the camera's JPEG undecoded, on [executor].
     * Returns false without calling [onDone] when no still capture is bound (the caller falls back to an analysis frame).
     */
    fun takeStill(executor: Executor, onDone: (Result<Shot>) -> Unit): Boolean {
        val capture = still ?: return false
        val facing = boundFacing ?: return false
        val target = CaptureRotation.degreesOf(capture.targetRotation)
        capture.takePicture(executor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                onDone(runCatching { image.use { Shot(jpegOf(it), it.imageInfo.rotationDegrees, target, facing) } })
            }

            override fun onError(exception: ImageCaptureException) {
                Log.w(TAG, "still capture failed", exception)
                onDone(Result.failure(exception))
            }
        })
        return true
    }

    fun bind(facing: CameraFacing) {
        requested = facing
        provider?.let { bindOn(it, facing); return }
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val p = future.get()
            provider = p
            requested?.let { bindOn(p, it) }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun bindOn(provider: ProcessCameraProvider, wanted: CameraFacing) {
        val available = CameraFacing.entries.filterTo(LinkedHashSet()) { hasCamera(provider, it) }
        val facing = CameraFacing.resolve(wanted, available)
        if (facing == null) {
            listener.onCameraError("No camera found")
            return
        }

        // Same 4:3 selector for both, so the analysis frame matches what PreviewView shows.
        val analysisSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(
                ResolutionStrategy(ANALYSIS_BOUND, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
            )
            .build()
        val previewSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .build()
        // Same 4:3 too (the still shows what the preview showed), as large as STILL_BOUND allows.
        val stillSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(
                ResolutionStrategy(STILL_BOUND, ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)
            )
            .build()

        fun preview() = Preview.Builder()
            .setResolutionSelector(previewSelector)
            .build()
            .also { it.setSurfaceProvider(previewView.surfaceProvider) }
        fun analysis() = ImageAnalysis.Builder()
            .setResolutionSelector(analysisSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { it.setAnalyzer(analysisExecutor, newAnalyzer()) }

        still = null
        analysis = null
        camera?.cameraInfo?.torchState?.removeObservers(owner)
        camera = null
        boundFacing = null
        try {
            provider.unbindAll()
            // Quality mode: the HAL's still-quality path (multi-frame noise reduction where it has one) and CameraX's
            // AF trigger before the shot; one Read still can afford the extra few hundred ms. JPEG 95 instead of that
            // mode's 100: about half the upload for no visible difference in text.
            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .setJpegQuality(STILL_JPEG_QUALITY)
                .setTargetRotation(captureRotation)
                .setResolutionSelector(stillSelector)
                .build()
            val frames = analysis()
            val bound = try {
                provider.bindToLifecycle(owner, selectorOf(facing), preview(), frames, capture).also {
                    still = capture
                    Log.i(TAG, "bound $facing with still capture ${capture.resolutionInfo?.resolution}")
                }
            } catch (e: IllegalArgumentException) {
                // Preview + analysis + JPEG is a guaranteed combination, but some HALs still refuse it: the Read
                // button then uses an analysis frame. Recognition matters more than the still.
                Log.w(TAG, "camera can't add still capture; binding preview + analysis only", e)
                provider.unbindAll()
                provider.bindToLifecycle(owner, selectorOf(facing), preview(), frames)
            }
            camera = bound
            analysis = frames
            boundFacing = facing
            bound.cameraInfo.torchState.observe(owner) { listener.onTorchChanged(it == TorchState.ON) }
            listener.onCameraBound(facing, available)
        } catch (e: IllegalArgumentException) {
            Log.e(TAG, "camera bind failed", e)
            listener.onCameraError("Camera bind failed: ${e.message}")
        }
    }

    private fun hasCamera(provider: ProcessCameraProvider, facing: CameraFacing): Boolean = try {
        provider.hasCamera(selectorOf(facing))
    } catch (_: CameraInfoUnavailableException) {
        false
    }

    private fun selectorOf(facing: CameraFacing): CameraSelector = when (facing) {
        CameraFacing.BACK -> CameraSelector.DEFAULT_BACK_CAMERA
        CameraFacing.FRONT -> CameraSelector.DEFAULT_FRONT_CAMERA
    }

    companion object {
        private const val TAG = "CameraBinder"
        private val ANALYSIS_BOUND = Size(1280, 960)
        /** Long side of the analysis frames we ask for (the size actually bound may differ). */
        const val ANALYSIS_LONG_SIDE = 1280
        /** Target for the Read text still: 12 MP 4:3, the closest smaller size if the sensor has no exact match. */
        private val STILL_BOUND = Size(4000, 3000)
        private const val STILL_JPEG_QUALITY = 95
        /** Metering region size (fraction of the frame) around a text mark's centre. */
        private const val MIN_METERING_SIZE = 0.1f
        private const val MAX_METERING_SIZE = 0.35f
        /** A Read focus lock returns to continuous AF by itself after this, if [cancelFocus] isn't called first. */
        private const val FOCUS_HOLD_S = 3L

        /** The JPEG bytes of an in-memory capture (re-encoded only if a device ever delivers another format). */
        private fun jpegOf(image: ImageProxy): ByteArray {
            if (image.format == ImageFormat.JPEG) {
                val buffer = image.planes[0].buffer
                buffer.rewind()
                return ByteArray(buffer.remaining()).also { buffer.get(it) }
            }
            val bitmap = image.toBitmap()
            try {
                val out = ByteArrayOutputStream(4 shl 20)
                bitmap.compress(Bitmap.CompressFormat.JPEG, STILL_JPEG_QUALITY, out)
                return out.toByteArray()
            } finally {
                bitmap.recycle()
            }
        }

        /** [image] (an RGBA analysis frame) as an upright bitmap, long side at most [maxLongSide]. */
        fun upright(image: ImageProxy, maxLongSide: Int): Bitmap {
            val source = image.toBitmap()
            val out = FrameEncoder.uprightBitmap(source, image.imageInfo.rotationDegrees, maxLongSide)
            if (out !== source) source.recycle()
            return out
        }
    }
}
