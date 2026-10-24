package pl.genor.glassvision

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.util.concurrent.Executor

/**
 * The Read button: freezes the frame the user aimed at and takes a still ([StillCapture]: focus on the text, hold
 * still, blur check), then opens a [ReadTextSheet] in its reading state, reads the still with
 * the backend's [TextReader] on [worker], shows the result; "Add language" translates the lines already read (no
 * second read). Target languages are remembered ([ReadTargets]). Only the open sheet's results are shown: a sheet
 * closed while reading drops them. Main thread only.
 */
class ReadTextFlow(
    private val context: Context,
    private val stills: StillCapture,
    /** Freeze/hints on the camera screen while the still is taken. */
    private val ui: CaptureUi,
    /** The live text marks to focus on (largest first; [TextMarks]). */
    private val marks: () -> List<NormRect>,
    private val reader: () -> TextReader?,
    private val targets: ReadTargets,
    private val worker: Executor,
    /** Called when [busy] changes (the Read button shows "Reading…"). */
    private val onBusyChanged: () -> Unit,
) {
    private val main = Handler(Looper.getMainLooper())
    private var session: Session? = null

    /** A still is being taken or read. */
    val busy: Boolean get() = session?.let { it.capturing || it.reading } == true

    fun start() {
        session?.let { if (it.capturing || it.sheet.sheet.isShowing) return }
        val textReader = reader() ?: return
        val s = Session(textReader)
        session = s
        s.sheet.sheet.onDismiss { if (session === s) session = null; onBusyChanged() }
        s.capturing = true
        onBusyChanged()
        // The sheet opens after the shot: one sliding up at the tap made people react and move the phone.
        ui.freeze()
        stills.take(marks(), ui) { result ->
            ui.unfreeze()
            s.capturing = false
            if (session !== s) return@take
            s.sheet.showReading()
            s.sheet.sheet.show(expanded = true)
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
        val sheet = ReadTextSheet(context, this)
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
            async({ s.decode(THUMBNAIL_LONG_SIDE) }) { r ->
                r.onSuccess { sheet.setStill(it, s.highRes) }
            }
            if (s.quality?.dark == true && ui.torchWouldHelp) sheet.showMessage(context.getString(R.string.read_too_dark))
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
                Triple(offered, wanted, textReader.readStill(s, wanted))
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
            val lines = read.lines.map { it.text }
            val from = read.lines.map { it.lang }
            async({ textReader.translate(lines, from, code) }) { r ->
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
    }

    private companion object {
        const val TAG = "ReadTextFlow"
        const val THUMBNAIL_LONG_SIDE = 720
    }
}
