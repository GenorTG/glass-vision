package pl.genor.glassvision

import android.content.Context
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraInfoUnavailableException
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
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
 * CameraX binding for preview + analysis on the back or front camera. [bind] (re)binds everything on the
 * chosen camera with fresh use cases and a fresh analyzer (so the frame schedule restarts), falling back to
 * the other camera if the device lacks it. Main thread only.
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

        val preview = Preview.Builder()
            .setResolutionSelector(previewSelector)
            .build()
            .also { it.setSurfaceProvider(previewView.surfaceProvider) }
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(analysisSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { it.setAnalyzer(analysisExecutor, newAnalyzer()) }

        try {
            provider.unbindAll()
            provider.bindToLifecycle(owner, selectorOf(facing), preview, analysis)
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

    private companion object {
        const val TAG = "CameraBinder"
        val ANALYSIS_BOUND = Size(1280, 960)
    }
}
