package pl.genor.glassvision

import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.CalendarContract
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.util.concurrent.Executor

/**
 * The Read button: takes a still ([StillCapture]), opens a [ReadTextSheet] in its reading state, reads the still with
 * the backend's [TextReader] on [worker], shows the result; "Add language" translates the blocks already read (no
 * second read). Target languages ([ReadTargets]) and the result view ([ReadViewChoice]) are remembered. Only the open
 * sheet's results are shown: a sheet closed while reading drops them. Main thread only.
 */
class ReadTextFlow(
    private val context: Context,
    private val stills: StillCapture,
    private val reader: () -> TextReader?,
    private val targets: ReadTargets,
    private val worker: Executor,
    /** Called when [busy] changes (the Read button shows "Reading…"). */
    private val onBusyChanged: () -> Unit,
    private val views: ReadViewChoice = ReadViewChoice.of(context),
) {
    private val main = Handler(Looper.getMainLooper())
    private var session: Session? = null

    /** A still is being taken or read. */
    val busy: Boolean get() = session?.let { it.capturing || it.reading } == true

    fun start() {
        session?.let { if (it.sheet.sheet.isShowing) return }
        val textReader = reader() ?: return
        val s = Session(textReader)
        session = s
        s.sheet.sheet.onDismiss { if (session === s) session = null; onBusyChanged() }
        s.sheet.showReading()
        s.sheet.sheet.show(expanded = true)
        s.capturing = true
        onBusyChanged()
        stills.take { result ->
            s.capturing = false
            if (session !== s) return@take
            result.fold(
                onSuccess = { still -> s.onStill(still) },
                onFailure = {
                    Log.w(TAG, "no still", it)
                    s.sheet.setStill(null, highRes = true)
                    s.sheet.showError(context.getString(R.string.read_capture_failed, it.message ?: it.javaClass.simpleName), canRetry = false)
                    onBusyChanged()
                },
            )
        }
    }

    fun dismiss() {
        session?.sheet?.sheet?.dismiss()
        session = null
    }

    private inner class Session(val textReader: TextReader) : ReadTextSheet.Callbacks {
        val sheet = ReadTextSheet(context, this, views.get())
        var capturing = false
        var reading = false
        private var still: Still? = null
        private var result: TextReading? = null

        private fun live() = session === this

        /** Runs [work] on the worker and [then] on the main thread while this sheet is still the open one. */
        private fun <T> async(work: () -> T, then: (Result<T>) -> Unit) {
            worker.execute {
                val r = runCatching(work)
                main.post { if (live()) then(r) }
            }
        }

        fun onStill(s: Still) {
            still = s
            async({ FrameEncoder.uprightBitmap(s.bitmap, 0, THUMBNAIL_LONG_SIDE) }) { r ->
                r.onSuccess { sheet.setStill(it, s.highRes) }
            }
            read()
        }

        private fun read() {
            val s = still ?: return
            reading = true
            onBusyChanged()
            sheet.showReading()
            async({
                val offered = runCatching { textReader.languages() }.getOrDefault(emptyList())
                val wanted = targets.forRead(offered)
                Triple(offered, wanted, textReader.read(s.bitmap, wanted))
            }) { r ->
                reading = false
                onBusyChanged()
                r.fold(
                    onSuccess = { (offered, wanted, reading) ->
                        result = reading
                        sheet.languages = offered
                        sheet.showResult(reading, wanted)
                    },
                    onFailure = {
                        Log.w(TAG, "read failed", it)
                        sheet.showError(it.message ?: it.javaClass.simpleName)
                    },
                )
            }
        }

        override fun onRetry() {
            if (!reading) read()
        }

        override fun onAddLanguage() {
            val offered = sheet.languages
            if (offered.isNotEmpty()) {
                sheet.pickLanguage(offered, ::addLanguage)
                return
            }
            async({ textReader.languages() }) { r ->
                r.fold(
                    onSuccess = {
                        sheet.languages = it
                        if (it.isEmpty()) sheet.showMessage(context.getString(R.string.read_no_languages, "—"))
                        else sheet.pickLanguage(it, ::addLanguage)
                    },
                    onFailure = { sheet.showMessage(context.getString(R.string.read_no_languages, it.message.orEmpty())) },
                )
            }
        }

        private fun addLanguage(code: String) {
            val read = result ?: return
            targets.add(code)
            sheet.setTranslation(code, ReadTextSheet.Translation.Pending)
            async({ textReader.translateBlocks(read.blocks, code) }) { r ->
                // Removed again while translating: nothing to show.
                if (code !in sheet.targets) return@async
                sheet.setTranslation(
                    code,
                    r.fold(
                        onSuccess = { ReadTextSheet.Translation.Ready(it) },
                        onFailure = { ReadTextSheet.Translation.Failed(it.message ?: it.javaClass.simpleName) },
                    ),
                )
            }
        }

        override fun onRemoveLanguage(code: String) {
            targets.remove(code)
            sheet.removeTranslation(code)
        }

        override fun onCopy(text: String) {
            val clipboard = context.getSystemService(ClipboardManager::class.java) ?: return
            clipboard.setPrimaryClip(ClipData.newPlainText(context.getString(R.string.read_title), text))
            // Android 13+ confirms copies itself.
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) sheet.showMessage(context.getString(R.string.read_copied))
        }

        override fun onShare(text: String) {
            val send = Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, text)
            context.startActivity(Intent.createChooser(send, context.getString(R.string.read_share)))
        }

        override fun onViewChanged(view: ReadView) = views.set(view)

        override fun onAction(detected: Detected) {
            val intent = when (detected.kind) {
                DetectedKind.PHONE -> Intent(Intent.ACTION_DIAL, Uri.parse("tel:" + detected.value))
                DetectedKind.URL -> Intent(Intent.ACTION_VIEW, Uri.parse(detected.value))
                DetectedKind.EMAIL -> Intent(Intent.ACTION_SENDTO, Uri.parse("mailto:" + detected.value))
                DetectedKind.DATE -> Intent(Intent.ACTION_INSERT, CalendarContract.Events.CONTENT_URI)
                    .putExtra(CalendarContract.EXTRA_EVENT_ALL_DAY, true)
                    .putExtra(CalendarContract.EXTRA_EVENT_BEGIN_TIME, ReadActions.dateMillis(detected.value))
            }
            try {
                context.startActivity(intent)
            } catch (_: ActivityNotFoundException) {
                sheet.showMessage(context.getString(R.string.read_no_app))
            }
        }
    }

    private companion object {
        const val TAG = "ReadTextFlow"
        const val THUMBNAIL_LONG_SIDE = 720
    }
}

/** Pure parts of the Read text actions. */
object ReadActions {
    /** Local midnight of an ISO date (yyyy-mm-dd) in epoch ms, for a calendar entry. */
    fun dateMillis(iso: String, zone: java.util.TimeZone = java.util.TimeZone.getDefault()): Long {
        val (y, m, d) = iso.split('-').map(String::toInt)
        return java.util.Calendar.getInstance(zone).apply {
            clear()
            set(y, m - 1, d)
        }.timeInMillis
    }
}
