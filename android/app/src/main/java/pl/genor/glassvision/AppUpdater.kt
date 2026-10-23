package pl.genor.glassvision

import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageInstaller
import android.os.Build
import android.os.Handler
import android.os.Looper
import java.io.File
import java.io.IOException
import java.security.MessageDigest
import java.util.concurrent.Executors

/**
 * Self-update from the glass-vision server: check `/app/version`, download `/app.apk` (size + SHA-256
 * verified), install through [PackageInstaller]. Network work runs on a background thread; every
 * callback is delivered on the main thread.
 *
 * The system only accepts the update if it is signed with the same key as the installed app. Android shows its
 * own confirmation unless this app installed itself last time (Android 12+, USER_ACTION_NOT_REQUIRED), in which
 * case later updates go through silently. Works against whichever server the user configured (token included).
 */
class AppUpdater(context: Context) {
    sealed interface CheckResult {
        data class Available(val info: UpdateInfo, val endpoint: ServerEndpoint) : CheckResult
        data object UpToDate : CheckResult
        data class Failed(val message: String) : CheckResult
    }

    private val appContext = context.applicationContext
    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "gv-updater").apply { isDaemon = true } }
    private val client = ServerHttp.client
    private val updatesDir = File(appContext.cacheDir, "updates")

    fun check(endpoint: ServerEndpoint, onResult: (CheckResult) -> Unit) {
        if (!endpoint.isConfigured) return onResult(CheckResult.Failed("no server configured"))
        val request = ServerHttp.get(endpoint, "/app/version")
            ?: return onResult(CheckResult.Failed("server URL is not ws:// or wss://"))
        worker.execute {
            val result = try {
                val body = client.newCall(request).execute().use { r ->
                    if (!r.isSuccessful) throw IOException("HTTP ${r.code}")
                    r.body?.string() ?: throw IOException("empty response")
                }
                val info = UpdateCheck.parse(body) ?: throw IOException("bad /app/version response")
                if (UpdateCheck.isUpdate(info, BuildConfig.VERSION_CODE, BuildConfig.APPLICATION_ID)) {
                    CheckResult.Available(info, endpoint)
                } else {
                    CheckResult.UpToDate
                }
            } catch (e: IOException) {
                CheckResult.Failed(e.message ?: e.javaClass.simpleName)
            }
            main.post { onResult(result) }
        }
    }

    /** Downloads and verifies the APK; [onProgress] gets 0..100. Re-uses an already verified download. */
    fun download(
        info: UpdateInfo,
        endpoint: ServerEndpoint,
        onProgress: (Int) -> Unit,
        onDone: (Result<File>) -> Unit,
    ) {
        worker.execute {
            val result = runCatching { downloadVerified(info, endpoint) { p -> main.post { onProgress(p) } } }
            main.post { onDone(result) }
        }
    }

    private fun downloadVerified(info: UpdateInfo, endpoint: ServerEndpoint, onProgress: (Int) -> Unit): File {
        updatesDir.mkdirs()
        val target = File(updatesDir, "glass-vision-${info.versionCode}.apk")
        if (target.length() == info.size && sha256(target) == info.sha256) return target
        updatesDir.listFiles()?.forEach { it.delete() } // drop older downloads

        val partial = File(updatesDir, "${target.name}.part")
        val digest = MessageDigest.getInstance("SHA-256")
        val request = ServerHttp.get(endpoint, info.url) ?: throw IOException("download: bad URL ${info.url}")
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IOException("download: HTTP ${response.code}")
            val body = response.body ?: throw IOException("download: empty body")
            var received = 0L
            var lastPercent = -1
            body.byteStream().use { input ->
                partial.outputStream().use { output ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        val n = input.read(buffer)
                        if (n < 0) break
                        output.write(buffer, 0, n)
                        digest.update(buffer, 0, n)
                        received += n
                        val percent = (received * 100 / info.size.coerceAtLeast(1)).toInt().coerceAtMost(100)
                        if (percent != lastPercent) {
                            lastPercent = percent
                            onProgress(percent)
                        }
                    }
                }
            }
            if (received != info.size) throw IOException("download: got $received bytes, expected ${info.size}")
        }
        val actual = digest.digest().toHex()
        if (actual != info.sha256) {
            partial.delete()
            throw IOException("download: SHA-256 mismatch (server rebuilt mid-download? try again)")
        }
        if (!partial.renameTo(target)) throw IOException("download: cannot move file into place")
        return target
    }

    /** False until the user allows "Install unknown apps" for this app. */
    fun canInstall(): Boolean = appContext.packageManager.canRequestPackageInstalls()

    /** Hands the APK to the system installer; the outcome arrives in [UpdateInstallReceiver]. */
    fun install(apk: File) {
        val installer = appContext.packageManager.packageInstaller
        val params = PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL).apply {
            setAppPackageName(appContext.packageName)
            setSize(apk.length())
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                setRequireUserAction(PackageInstaller.SessionParams.USER_ACTION_NOT_REQUIRED)
            }
        }
        val sessionId = installer.createSession(params)
        try {
            installer.openSession(sessionId).use { session ->
                session.openWrite("base.apk", 0, apk.length()).use { out ->
                    apk.inputStream().use { it.copyTo(out) }
                    session.fsync(out)
                }
                val mutable = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) PendingIntent.FLAG_MUTABLE else 0
                val callback = PendingIntent.getBroadcast(
                    appContext,
                    sessionId,
                    Intent(appContext, UpdateInstallReceiver::class.java),
                    PendingIntent.FLAG_UPDATE_CURRENT or mutable, // the installer fills in the status extras
                )
                session.commit(callback.intentSender)
            }
        } catch (e: Exception) {
            installer.abandonSession(sessionId)
            throw e
        }
    }

    fun shutdown() {
        worker.shutdownNow()
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val n = input.read(buffer)
                if (n < 0) break
                digest.update(buffer, 0, n)
            }
        }
        return digest.digest().toHex()
    }

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }
}
