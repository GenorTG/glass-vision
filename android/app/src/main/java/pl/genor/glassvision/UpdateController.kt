package pl.genor.glassvision

import android.content.Intent
import android.net.Uri
import android.os.SystemClock
import android.provider.Settings
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.progressindicator.LinearProgressIndicator
import java.io.File

/**
 * The update UI around [AppUpdater]: offer → download with progress → "allow installs from this app"
 * detour if needed → system installer. Main thread only; owned by [MainActivity].
 */
class UpdateController(
    private val activity: AppCompatActivity,
    private val endpoint: () -> ServerEndpoint,
) {
    private val updater = AppUpdater(activity)
    private var busy = false
    /** Downloaded APK waiting for the user to grant "Install unknown apps" in system settings. */
    private var pendingApk: File? = null

    /**
     * Automatic check on every (re)connect, at most every [AUTO_CHECK_INTERVAL_MS]; silent unless an update
     * exists. Android keeps the process alive for hours, so "once per process" would mean "never again".
     */
    fun checkQuietly() {
        val now = SystemClock.elapsedRealtime()
        if (lastAutoCheck == 0L || now - lastAutoCheck >= AUTO_CHECK_INTERVAL_MS) {
            lastAutoCheck = now
            check(manual = false)
        }
    }

    /** "Check for updates" button: always reports the outcome. */
    fun checkNow() = check(manual = true)

    /** Call from onResume: finishes an install that was waiting for the permission. */
    fun onResume() {
        val apk = pendingApk ?: return
        if (updater.canInstall()) {
            pendingApk = null
            install(apk)
        }
    }

    fun shutdown() = updater.shutdown()

    private fun check(manual: Boolean) {
        if (busy) return
        busy = true
        updater.check(endpoint()) { result ->
            busy = false
            if (activity.isFinishing) return@check
            when (result) {
                is AppUpdater.CheckResult.Available -> offer(result.info, result.endpoint)
                AppUpdater.CheckResult.UpToDate ->
                    if (manual) toast(activity.getString(R.string.update_up_to_date, BuildConfig.VERSION_NAME))
                is AppUpdater.CheckResult.Failed -> {
                    Log.w(TAG, "update check failed: ${result.message}")
                    if (!manual) lastAutoCheck = 0L // retry on the next connect instead of in 10 min
                    if (manual) toast(activity.getString(R.string.update_check_failed, result.message))
                }
            }
        }
    }

    private fun offer(info: UpdateInfo, server: ServerEndpoint) {
        MaterialAlertDialogBuilder(activity)
            .setTitle(R.string.update_available_title)
            .setMessage(
                activity.getString(
                    R.string.update_available_message,
                    info.versionName,
                    BuildConfig.VERSION_NAME,
                    info.size / 1_048_576.0,
                )
            )
            .setPositiveButton(R.string.update_install) { _, _ -> download(info, server) }
            .setNegativeButton(R.string.update_later, null)
            .show()
    }

    private fun download(info: UpdateInfo, server: ServerEndpoint) {
        busy = true
        val bar = LinearProgressIndicator(activity).apply {
            max = 100
            val pad = (24 * resources.displayMetrics.density).toInt()
            setPadding(pad, pad, pad, pad)
        }
        val dialog = MaterialAlertDialogBuilder(activity)
            .setTitle(activity.getString(R.string.update_downloading, info.versionName))
            .setView(bar)
            .setCancelable(false)
            .show()
        updater.download(info, server, onProgress = { bar.progress = it }) { result ->
            busy = false
            if (activity.isFinishing) return@download
            dialog.dismiss()
            result.fold(
                onSuccess = ::install,
                onFailure = { toast(activity.getString(R.string.update_failed, it.message ?: it.javaClass.simpleName)) },
            )
        }
    }

    private fun install(apk: File) {
        if (!updater.canInstall()) {
            pendingApk = apk
            MaterialAlertDialogBuilder(activity)
                .setTitle(R.string.update_permission_title)
                .setMessage(R.string.update_permission_message)
                .setPositiveButton(R.string.update_open_settings) { _, _ ->
                    activity.startActivity(
                        Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:${activity.packageName}"))
                    )
                }
                .setNegativeButton(R.string.update_later) { _, _ -> pendingApk = null }
                .show()
            return
        }
        try {
            updater.install(apk)
        } catch (e: Exception) {
            toast(activity.getString(R.string.update_failed, e.message ?: e.javaClass.simpleName))
        }
    }

    private fun toast(text: String) = Toast.makeText(activity, text, Toast.LENGTH_LONG).show()

    private companion object {
        const val TAG = "GlassVisionUpdate"
        const val AUTO_CHECK_INTERVAL_MS = 10 * 60 * 1000L

        /** elapsedRealtime of the last automatic check, process-wide (0 = never). */
        var lastAutoCheck = 0L
    }
}
