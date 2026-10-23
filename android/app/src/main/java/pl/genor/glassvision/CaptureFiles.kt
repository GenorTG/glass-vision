package pl.genor.glassvision

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/*
 * Where photos and server-produced files go on the phone. Pure and JVM-testable; MediaSaver does the writing.
 */

/** Shared media collection a file is saved into. [root] = its top-level folder on shared storage. */
enum class SaveCollection(val root: String) {
    IMAGES("Pictures"),
    DOWNLOADS("Download"),
}

/** One file to write: [subfolder] is below the collection root, e.g. `GlassVision/20261023-101500-123`. */
data class SaveTarget(
    val displayName: String,
    val mimeType: String,
    val collection: SaveCollection,
    val subfolder: String,
) {
    /** MediaStore RELATIVE_PATH (API 29+), e.g. `Pictures/GlassVision/<id>`. */
    val relativePath: String get() = "${collection.root}/$subfolder"
    val isImage: Boolean get() = collection == SaveCollection.IMAGES
}

object CaptureFiles {
    const val APP_FOLDER = "GlassVision"
    private val UNSAFE = Regex("[^A-Za-z0-9._-]")

    /** The photo button's file: `Pictures/GlassVision/glassvision_yyyyMMdd_HHmmss.jpg`. */
    fun photo(now: Date): SaveTarget = SaveTarget(
        displayName = "glassvision_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(now) + ".jpg",
        mimeType = "image/jpeg",
        collection = SaveCollection.IMAGES,
        subfolder = APP_FOLDER,
    )

    /**
     * Targets for a `command_result.files` list, in order. Images go to the gallery, everything else (json…) to
     * Downloads, both under `GlassVision/<capture id>/`; the id comes from the url (`/captures/<id>/<name>`).
     */
    fun plan(files: List<ServerMessage.CommandResult.ResultFile>, fallbackId: String): List<SaveTarget> = files.map { f ->
        val name = safeName(f.name)
        val mime = mimeOf(name)
        SaveTarget(
            displayName = name,
            mimeType = mime,
            collection = if (mime.startsWith("image/")) SaveCollection.IMAGES else SaveCollection.DOWNLOADS,
            subfolder = "$APP_FOLDER/${captureId(f.url) ?: safeName(fallbackId)}",
        )
    }

    /** `/captures/20261023-101500-123/photo.jpg` → `20261023-101500-123`; null without a parent folder. */
    fun captureId(url: String): String? {
        val segments = url.substringBefore('?').substringBefore('#').split('/').filter { it.isNotEmpty() }
        if (segments.size < 2) return null
        return safeName(segments[segments.size - 2]).takeIf { it.isNotEmpty() && it != "_" }
    }

    /** File name without any path, limited to a portable character set (never `..` or a separator). */
    fun safeName(name: String): String {
        val base = name.replace('\\', '/').substringAfterLast('/').trim()
        val clean = base.replace(UNSAFE, "_").trimStart('.')
        return clean.ifEmpty { "file" }.take(MAX_NAME)
    }

    fun mimeOf(name: String): String = when (name.substringAfterLast('.', "").lowercase()) {
        "jpg", "jpeg" -> "image/jpeg"
        "png" -> "image/png"
        "webp" -> "image/webp"
        "json" -> "application/json"
        "txt" -> "text/plain"
        else -> "application/octet-stream"
    }

    private const val MAX_NAME = 120
}

/** Integer crop rect (JVM-testable). */
data class PixelRect(val left: Int, val top: Int, val right: Int, val bottom: Int) {
    val width: Int get() = right - left
    val height: Int get() = bottom - top
}

object PhotoFrame {
    /** Portrait 3:4: CameraBinder binds preview and analysis with the same 4:3 selector. */
    const val DEFAULT_FRAME_W = 3
    const val DEFAULT_FRAME_H = 4

    /**
     * The part of a [viewW]×[viewH] preview bitmap that shows the camera image ([frameW]×[frameH] under FIT_CENTER),
     * so the saved photo has no letterbox bars. Falls back to the whole view for degenerate input.
     */
    fun crop(viewW: Int, viewH: Int, frameW: Int, frameH: Int): PixelRect {
        val r = BoxMapper.fitCenterRect(frameW, frameH, viewW, viewH)
        val left = Math.round(r.left).coerceIn(0, viewW)
        val top = Math.round(r.top).coerceIn(0, viewH)
        val right = Math.round(r.right).coerceIn(left, viewW)
        val bottom = Math.round(r.bottom).coerceIn(top, viewH)
        return if (right - left < 1 || bottom - top < 1) PixelRect(0, 0, viewW, viewH) else PixelRect(left, top, right, bottom)
    }
}
