package pl.genor.glassvision

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.annotation.RequiresApi
import java.io.File
import java.io.IOException
import java.io.OutputStream

/**
 * Writes a [SaveTarget] to shared storage. Blocking: call from a worker thread.
 *
 * - API 29+: MediaStore (Images or Downloads) with RELATIVE_PATH, pending until fully written; no permission.
 * - API 26–28: images as files under the public Pictures folder, then registered with MediaStore so the gallery
 *   sees them (needs WRITE_EXTERNAL_STORAGE, requested by the activity); other files go to the app-specific
 *   external Downloads folder (no shared Downloads collection before 29).
 */
class MediaSaver(context: Context) {
    private val appContext = context.applicationContext
    private val resolver = appContext.contentResolver

    /** [write] fills the file; returns a Uri an ACTION_VIEW intent can open (for images). */
    fun save(target: SaveTarget, write: (OutputStream) -> Unit): Uri =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) saveScoped(target, write) else saveLegacy(target, write)

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun saveScoped(target: SaveTarget, write: (OutputStream) -> Unit): Uri {
        val collection = when (target.collection) {
            SaveCollection.IMAGES -> MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
            SaveCollection.DOWNLOADS -> MediaStore.Downloads.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        }
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, target.displayName)
            put(MediaStore.MediaColumns.MIME_TYPE, target.mimeType)
            put(MediaStore.MediaColumns.RELATIVE_PATH, target.relativePath + "/")
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        val uri = resolver.insert(collection, values) ?: throw IOException("cannot create ${target.displayName}")
        try {
            val out = resolver.openOutputStream(uri) ?: throw IOException("cannot open ${target.displayName}")
            out.use(write)
            resolver.update(uri, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null)
        } catch (e: Exception) {
            resolver.delete(uri, null, null)
            throw e
        }
        return uri
    }

    @Suppress("DEPRECATION") // DATA + public directories are the API 26–28 way.
    private fun saveLegacy(target: SaveTarget, write: (OutputStream) -> Unit): Uri {
        val dir = when (target.collection) {
            SaveCollection.IMAGES -> File(Environment.getExternalStorageDirectory(), target.relativePath)
            SaveCollection.DOWNLOADS -> File(
                appContext.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS) ?: throw IOException("no external storage"),
                target.subfolder,
            )
        }
        if (!dir.isDirectory && !dir.mkdirs()) throw IOException("cannot create $dir")
        val file = uniqueFile(dir, target.displayName)
        try {
            file.outputStream().use(write)
        } catch (e: Exception) {
            file.delete()
            throw e
        }
        if (!target.isImage) return Uri.fromFile(file)
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DATA, file.absolutePath)
            put(MediaStore.MediaColumns.DISPLAY_NAME, file.name)
            put(MediaStore.MediaColumns.MIME_TYPE, target.mimeType)
        }
        return resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: Uri.fromFile(file)
    }

    /** `name.ext`, else `name (1).ext`, `name (2).ext`… (MediaStore does this itself on 29+). */
    private fun uniqueFile(dir: File, name: String): File {
        val stem = name.substringBeforeLast('.', name)
        val ext = name.substringAfterLast('.', "").let { if (it.isEmpty()) "" else ".$it" }
        var candidate = File(dir, name)
        var n = 1
        while (candidate.exists()) candidate = File(dir, "$stem ($n)$ext").also { n++ }
        return candidate
    }
}
