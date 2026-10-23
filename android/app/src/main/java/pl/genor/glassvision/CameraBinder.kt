package pl.genor.glassvision

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraInfoUnavailableException
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executor

/**
 * CameraX binding for preview + analysis (+ a high-resolution still for Read text) on the back or front camera.
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
    }

    private var provider: ProcessCameraProvider? = null
    private var requested: CameraFacing? = null
    private var still: ImageCapture? = null

    /** Size of the bound still capture (sensor orientation), or null when only preview + analysis are bound. */
    val stillResolution: Size? get() = still?.resolutionInfo?.resolution

    /**
     * Takes one high-resolution still and delivers it upright (rotation applied; front camera unmirrored, like the
     * analysis frames), at most [STILL_MAX_LONG_SIDE] px on the long side, on [executor]. Returns false without
     * calling [onDone] when no still capture is bound (the caller falls back to an analysis frame).
     */
    fun takeStill(executor: Executor, onDone: (Result<Bitmap>) -> Unit): Boolean {
        val capture = still ?: return false
        capture.takePicture(executor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                onDone(runCatching { image.use { upright(it, STILL_MAX_LONG_SIDE) } })
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
        try {
            provider.unbindAll()
            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setResolutionSelector(stillSelector)
                .build()
            try {
                provider.bindToLifecycle(owner, selectorOf(facing), preview(), analysis(), capture)
                still = capture
                Log.i(TAG, "bound $facing with still capture ${capture.resolutionInfo?.resolution}")
            } catch (e: IllegalArgumentException) {
                // Preview + analysis + JPEG is a guaranteed combination, but some HALs still refuse it: the Read
                // button then uses an analysis frame. Recognition matters more than the still.
                Log.w(TAG, "camera can't add still capture; binding preview + analysis only", e)
                provider.unbindAll()
                provider.bindToLifecycle(owner, selectorOf(facing), preview(), analysis())
            }
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
        /** Target for the Read text still: 12 MP 4:3, the closest smaller size if the sensor has no exact match. */
        private val STILL_BOUND = Size(4000, 3000)
        const val STILL_MAX_LONG_SIDE = 4000

        /** [image] (RGBA analysis frame or JPEG capture) as an upright bitmap, long side at most [maxLongSide]. */
        fun upright(image: ImageProxy, maxLongSide: Int): Bitmap {
            // JPEG: toBitmap() decodes the pixels as stored (EXIF ignored) and rotationDegrees says how to turn them,
            // exactly like an analysis frame. In-memory captures are never mirrored, so the front camera matches too.
            val source = image.toBitmap()
            val out = FrameEncoder.uprightBitmap(source, image.imageInfo.rotationDegrees, maxLongSide)
            if (out !== source) source.recycle()
            return out
        }
    }
}
