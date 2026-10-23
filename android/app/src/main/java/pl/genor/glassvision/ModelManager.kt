package pl.genor.glassvision

import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import javax.net.ssl.SSLException

/** One downloadable model file: where it comes from and what it must be. */
data class ModelSpec(
    val id: String,
    val fileName: String,
    val title: String,
    val url: String,
    val bytes: Long,
    val sha256: String,
)

val LocalModel.spec: ModelSpec get() = ModelSpec(name, fileName, title, url, bytes, sha256)

/** A model file on the phone (Settings → "Models on this phone"). */
data class StoredModel(val id: String, val title: String, val bytes: Long)

/** What the settings sheet needs from the model store (snapshot tests pass a fake). */
interface ModelShelf {
    fun downloaded(): List<StoredModel>

    /** False if there was nothing to delete. */
    fun delete(id: String): Boolean
}

/** Opens a URL for reading, optionally from a byte offset. Swappable for a fake in JVM tests. */
fun interface Downloader {
    /** [resumed] = the body starts at the requested offset (HTTP 206); false = it starts at byte 0. */
    class Body(val stream: InputStream, val resumed: Boolean)

    /** Throws [IOException] (incl. [HttpStatusException]) on failure. */
    fun open(url: String, offset: Long): Body
}

class HttpStatusException(val code: Int) : IOException("HTTP $code")

/** A model could not be made available; [message] is meant for the user (shown on the pipeline's chip/card). */
class ModelUnavailableException(message: String, cause: Throwable? = null) : IOException(message, cause)

/** [Downloader] over OkHttp; a Range request when resuming. */
class OkHttpDownloader(
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build(),
) : Downloader {
    override fun open(url: String, offset: Long): Downloader.Body {
        val request = Request.Builder().url(url).apply { if (offset > 0) header("Range", "bytes=$offset-") }.build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            response.close()
            throw HttpStatusException(response.code)
        }
        val body = response.body ?: run {
            response.close()
            throw IOException("empty response")
        }
        return Downloader.Body(body.byteStream(), resumed = response.code == 206)
    }
}

/**
 * Model files for the on-device engine in app-specific storage ([dir], e.g. `files/models`). [ensure] returns a
 * verified file, downloading it first if needed: into `<name>.part` (resumed with a Range request after an
 * interruption), then size + sha256 checked, then renamed into place, so a half-written file is never used.
 * One download per model at a time (a second caller waits for the first). Blocking: call off the main thread.
 */
class ModelManager(
    private val dir: File,
    private val downloader: Downloader,
    /** Every model the app knows (for [downloaded]). */
    private val known: List<ModelSpec> = LocalModel.entries.map { it.spec },
    /** Free bytes where [dir] lives (fake in tests). */
    private val freeBytes: (File) -> Long = { it.usableSpace },
) : ModelShelf {
    private val locks = ConcurrentHashMap<String, Any>()

    fun file(model: ModelSpec): File = File(dir, model.fileName)

    private fun part(model: ModelSpec): File = File(dir, model.fileName + ".part")

    /** Downloaded and of the right size (the hash was checked when it was downloaded). */
    fun isReady(model: ModelSpec): Boolean = file(model).let { it.isFile && it.length() == model.bytes }

    /**
     * The model's file, downloading it if needed. [onProgress] gets (bytes so far, total). Throws
     * [ModelUnavailableException] with a readable message.
     */
    fun ensure(model: ModelSpec, onProgress: (Long, Long) -> Unit = { _, _ -> }): File =
        synchronized(locks.getOrPut(model.fileName) { Any() }) {
            val target = file(model)
            if (isReady(model)) return target
            if (target.exists()) target.delete() // wrong size: a leftover of another version
            val parent = target.parentFile ?: throw ModelUnavailableException("Invalid model path")
            if (!parent.isDirectory && !parent.mkdirs()) {
                throw ModelUnavailableException("Can't create the model folder")
            }
            try {
                download(model, onProgress)
            } catch (e: ModelUnavailableException) {
                throw e
            } catch (e: IOException) {
                throw ModelUnavailableException(describe(e, model), e)
            }
            target
        }

    private fun download(model: ModelSpec, onProgress: (Long, Long) -> Unit) {
        val part = part(model)
        if (part.length() > model.bytes) part.delete()
        var have = part.length()
        val missing = model.bytes - have
        if (missing > 0 && freeBytes(dir) < missing + FREE_SPACE_MARGIN) {
            throw ModelUnavailableException("Not enough free space: ${model.title} needs ${mb(missing)}")
        }
        if (have < model.bytes) {
            val body = downloader.open(model.url, have)
            if (!body.resumed) have = 0 // server sent the whole file: start over
            body.stream.use { input ->
                FileOutputStream(part, have > 0).use { out ->
                    val buf = ByteArray(BUFFER)
                    var done = have
                    onProgress(done, model.bytes)
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        if (done + n > model.bytes) {
                            part.delete()
                            throw ModelUnavailableException("Model download failed: ${model.title} is larger than expected")
                        }
                        out.write(buf, 0, n)
                        done += n
                        onProgress(done, model.bytes)
                    }
                }
            }
        }
        if (part.length() != model.bytes) {
            // Ended early (connection dropped): keep the part for the next attempt to resume.
            throw ModelUnavailableException("Model download interrupted (${mb(part.length())} of ${mb(model.bytes)}) — it resumes next time")
        }
        if (sha256(part) != model.sha256) {
            part.delete()
            throw ModelUnavailableException("Model download corrupted (checksum mismatch) — try again")
        }
        if (!part.renameTo(file(model))) throw ModelUnavailableException("Can't save the model file")
    }

    override fun downloaded(): List<StoredModel> = known.mapNotNull { m ->
        val f = file(m)
        if (f.isFile) StoredModel(m.id, m.title, f.length()) else null
    }

    override fun delete(id: String): Boolean {
        val m = known.firstOrNull { it.id == id } ?: return false
        return synchronized(locks.getOrPut(m.fileName) { Any() }) {
            val had = file(m).exists() || part(m).exists()
            file(m).delete()
            part(m).delete()
            had
        }
    }

    private fun describe(e: IOException, model: ModelSpec): String = when (e) {
        is UnknownHostException, is ConnectException -> "No internet — model download failed"
        is SocketTimeoutException -> "Download timed out — model download failed"
        is SSLException -> "Secure connection failed — model download failed"
        is HttpStatusException -> "Model download failed (HTTP ${e.code})"
        else -> "Model download failed: ${e.message ?: e.javaClass.simpleName} (${model.title})"
    }

    companion object {
        private const val BUFFER = 64 * 1024
        /** Keep this much space free beyond the model itself. */
        private const val FREE_SPACE_MARGIN = 16L * 1024 * 1024

        fun sha256(file: File): String {
            val md = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buf = ByteArray(BUFFER)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    md.update(buf, 0, n)
                }
            }
            return md.digest().joinToString("") { "%02x".format(it) }
        }

        /** "7.3 MB" (one decimal). */
        fun mb(bytes: Long): String = String.format(Locale.US, "%.1f MB", bytes / 1_048_576.0)
    }
}
