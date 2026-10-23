package pl.genor.glassvision

import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executor

/**
 * Saves the files of a `command_result` (docs/PROTOCOL.md, `files`) via [MediaSaver]: images into the gallery, the
 * rest into Downloads, both under `GlassVision/<capture id>/`. Server files are downloaded from the configured
 * server with its access token; files the on-device engine produced (`file:` URLs) are copied, but only from
 * inside [localRoot] (null in server mode, so a server can never point the app at its own private files).
 * Pipeline-agnostic. Network + disk on [io]; [onDone] on the main thread.
 */
class ResultFileSaver(
    private val saver: MediaSaver,
    private val io: Executor,
) {
    /** [firstImage] = what an "Open" action shows; [error] = the first failure, if any. */
    data class Outcome(val saved: Int, val failed: Int, val firstImage: Uri?, val error: String?)

    private val main = Handler(Looper.getMainLooper())

    fun save(
        endpoint: ServerEndpoint,
        files: List<ServerMessage.CommandResult.ResultFile>,
        localRoot: File? = null,
        onDone: (Outcome) -> Unit,
    ) {
        val fallbackId = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val targets = CaptureFiles.plan(files, fallbackId)
        io.execute {
            var saved = 0
            var firstImage: Uri? = null
            var error: String? = null
            files.zip(targets).forEach { (file, target) ->
                try {
                    val uri = if (localRoot != null) copyLocal(localRoot, file, target) else download(endpoint, file, target)
                    saved++
                    if (target.isImage && firstImage == null) firstImage = uri
                } catch (e: Exception) {
                    Log.w(TAG, "saving ${file.url} failed", e)
                    if (error == null) error = "${file.name}: ${e.message ?: e.javaClass.simpleName}"
                }
            }
            val outcome = Outcome(saved, files.size - saved, firstImage, error)
            main.post { onDone(outcome) }
        }
    }

    private fun download(endpoint: ServerEndpoint, file: ServerMessage.CommandResult.ResultFile, target: SaveTarget): Uri {
        val request = ServerHttp.get(endpoint, file.url) ?: throw IOException("bad URL")
        return ServerHttp.client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IOException("HTTP ${response.code}")
            val body = response.body ?: throw IOException("empty response")
            saver.save(target) { out -> body.byteStream().use { it.copyTo(out) } }
        }
    }

    private fun copyLocal(root: File, file: ServerMessage.CommandResult.ResultFile, target: SaveTarget): Uri {
        val source = CaptureFiles.localSource(file.url, root) ?: throw IOException("not a capture file")
        return saver.save(target) { out -> source.inputStream().use { it.copyTo(out) } }
    }

    private companion object {
        const val TAG = "ResultFileSaver"
    }
}
