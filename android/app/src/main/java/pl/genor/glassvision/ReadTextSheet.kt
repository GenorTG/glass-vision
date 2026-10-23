package pl.genor.glassvision

import android.content.Context
import android.graphics.Bitmap
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.progressindicator.CircularProgressIndicator
import com.google.android.material.snackbar.Snackbar

/**
 * Read text results: the still with its line boxes, then reading… / error + retry / no text found, or the recognized
 * text as readable blocks followed by one section per translation and "Add language". Not drawn on the camera: plain
 * selectable text with copy and share. Plain data in, [Callbacks] out; [ReadTextFlow] does the work. Main thread only.
 */
class ReadTextSheet(private val context: Context, private val callbacks: Callbacks) {

    interface Callbacks {
        fun onRetry()
        fun onAddLanguage()
        fun onRemoveLanguage(code: String)
        fun onCopy(text: String)
        fun onShare(text: String)
    }

    /** One target language's section. */
    sealed interface Translation {
        data object Pending : Translation
        /** Aligned with the lines; null = untranslated (shows the original line). */
        data class Ready(val entries: List<String?>) : Translation
        data class Failed(val message: String) : Translation
    }

    val sheet = Sheet(context)
    private val inflater = LayoutInflater.from(context)
    private val dp = context.resources.displayMetrics.density
    private val body = inflater.inflate(R.layout.sheet_read_text_body, null as ViewGroup?, false)
    private val thumbnail: ReadThumbnailView = body.findViewById(R.id.read_thumbnail)
    private val status: View = body.findViewById(R.id.read_status)
    private val progress: CircularProgressIndicator = body.findViewById(R.id.read_progress)
    private val statusIcon: ImageView = body.findViewById(R.id.read_status_icon)
    private val statusText: TextView = body.findViewById(R.id.read_status_text)
    private val statusDetail: TextView = body.findViewById(R.id.read_status_detail)
    private val retry: MaterialButton = body.findViewById(R.id.read_retry)
    private val sections: LinearLayout = body.findViewById(R.id.read_sections)
    private val addLanguage: MaterialButton = body.findViewById(R.id.read_add_language)

    private var highRes = true
    private var reading: TextReading? = null
    private val translations = LinkedHashMap<String, Translation>()

    /** Names for language codes ([TextReader.languages]); codes without one use the phone's name. */
    var languages: List<TranslationLanguage> = emptyList()
        set(value) {
            field = value
            if (reading != null) renderResult()
        }

    /** Target codes shown, in order. */
    val targets: List<String> get() = translations.keys.toList()

    init {
        sheet.setTitle(context.getString(R.string.read_title))
        sheet.setBody(body)
        retry.setOnClickListener { callbacks.onRetry() }
        addLanguage.setOnClickListener { callbacks.onAddLanguage() }
    }

    /** The picture being read (a downscaled copy is enough); [highRes] = false for the analysis-frame fallback. */
    fun setStill(picture: Bitmap?, highRes: Boolean) {
        this.highRes = highRes
        thumbnail.setImageBitmap(picture)
        thumbnail.visibility = if (picture != null) View.VISIBLE else View.GONE
    }

    fun showReading() {
        reading = null
        translations.clear()
        thumbnail.boxes = emptyList()
        sheet.setSubtitle(null)
        showStatus(busy = true, icon = null, title = R.string.read_reading, detail = context.getString(R.string.read_reading_detail))
    }

    fun showError(message: String, canRetry: Boolean = true) {
        reading = null
        sheet.setSubtitle(null)
        showStatus(busy = false, icon = R.drawable.ic_error_dot, title = R.string.read_failed, detail = message, retry = canRetry)
    }

    /** [targets] = what was asked for, in order; each gets its translation, its error, or (missing) an error. */
    fun showResult(result: TextReading, targets: List<String>) {
        reading = result
        translations.clear()
        for (code in (targets + result.translations.keys + result.errors.keys).distinct()) {
            translations[code] = result.translations[code]?.let { Translation.Ready(it) }
                ?: Translation.Failed(result.errors[code] ?: "not translated")
        }
        thumbnail.boxes = result.lines.map { it.box }
        sheet.setSubtitle(subtitle(result))
        if (result.lines.isEmpty()) {
            showStatus(
                busy = false, icon = R.drawable.ic_read_text, title = R.string.read_no_text,
                detail = context.getString(R.string.read_no_text_detail),
            )
            return
        }
        renderResult()
    }

    fun setTranslation(code: String, translation: Translation) {
        if (reading == null) return
        translations[code] = translation
        renderResult()
    }

    fun removeTranslation(code: String) {
        if (translations.remove(code) != null && reading != null) renderResult()
    }

    /** Language picker for "Add language": [offered] minus what is shown already. */
    fun pickLanguage(offered: List<TranslationLanguage>, onPick: (String) -> Unit) {
        val choices = offered.filter { it.code !in translations }
        if (choices.isEmpty()) return
        MaterialAlertDialogBuilder(context)
            .setTitle(R.string.read_add_language_title)
            .setItems(choices.map { it.name }.toTypedArray()) { _, which -> onPick(choices[which].code) }
            .setNegativeButton(R.string.command_cancel, null)
            .show()
    }

    fun showMessage(message: String) {
        if (sheet.isShowing) Snackbar.make(sheet.view, message, Snackbar.LENGTH_SHORT).show()
    }

    private fun subtitle(result: TextReading): String {
        val parts = mutableListOf<String>()
        if (result.lines.isNotEmpty()) {
            parts += context.resources.getQuantityString(R.plurals.read_lines, result.lines.size, result.lines.size)
        }
        result.script?.takeIf { it.isNotBlank() }?.let { parts += it.replaceFirstChar(Char::titlecase) }
        if (!highRes) parts += context.getString(R.string.read_low_res)
        return parts.joinToString(" · ")
    }

    private fun showStatus(busy: Boolean, icon: Int?, title: Int, detail: String?, retry: Boolean = false) {
        status.visibility = View.VISIBLE
        sections.visibility = View.GONE
        addLanguage.visibility = View.GONE
        progress.visibility = if (busy) View.VISIBLE else View.GONE
        statusIcon.visibility = if (icon != null) View.VISIBLE else View.GONE
        icon?.let(statusIcon::setImageResource)
        // The error dot carries its own colours; other icons follow the text.
        statusIcon.imageTintList = if (icon == R.drawable.ic_error_dot) {
            null
        } else {
            ContextCompat.getColorStateList(context, R.color.gv_on_surface_variant)
        }
        statusText.setText(title)
        statusDetail.text = detail
        statusDetail.visibility = if (detail.isNullOrBlank()) View.GONE else View.VISIBLE
        this.retry.visibility = if (retry) View.VISIBLE else View.GONE
    }

    private fun renderResult() {
        val result = reading ?: return
        if (result.lines.isEmpty()) return
        status.visibility = View.GONE
        sections.visibility = View.VISIBLE
        addLanguage.visibility = View.VISIBLE
        sections.removeAllViews()

        val source = ReadTextFormat.sourceLanguage(result)?.let { languageName(it, languages) }
        val original = section(
            title = source?.let { context.getString(R.string.read_original_in, it) } ?: context.getString(R.string.read_original),
            icon = R.drawable.ic_read_text,
        )
        fillBlocks(original, result) { result.lines[it].text }
        val copyText = ReadTextFormat.original(result)
        original.findViewById<View>(R.id.section_copy).setOnClickListener { callbacks.onCopy(copyText) }
        original.findViewById<View>(R.id.section_share).apply {
            visibility = View.VISIBLE
            setOnClickListener { callbacks.onShare(shareText(result)) }
        }
        (original.layoutParams as LinearLayout.LayoutParams).topMargin = (12 * dp).toInt()
        sections.addView(original)

        for ((code, translation) in translations) {
            val name = languageName(code, languages)
            val view = section(title = name, icon = R.drawable.ic_translate)
            view.findViewById<ImageButton>(R.id.section_remove).apply {
                visibility = View.VISIBLE
                contentDescription = context.getString(R.string.read_remove_language, name)
                setOnClickListener { callbacks.onRemoveLanguage(code) }
            }
            val copy = view.findViewById<View>(R.id.section_copy)
            when (translation) {
                Translation.Pending -> {
                    copy.visibility = View.GONE
                    sectionState(view, busy = true, error = false, context.getString(R.string.read_translating))
                }
                is Translation.Failed -> {
                    copy.visibility = View.GONE
                    sectionState(view, busy = false, error = true, context.getString(R.string.read_translation_failed, translation.message))
                }
                is Translation.Ready -> if (ReadTextFormat.isUnchanged(translation.entries)) {
                    copy.visibility = View.GONE
                    sectionState(view, busy = false, error = false, context.getString(R.string.read_unchanged, name))
                } else {
                    fillBlocks(view, result) { translation.entries.getOrNull(it) ?: result.lines[it].text }
                    val text = ReadTextFormat.translated(result, translation.entries)
                    copy.setOnClickListener { callbacks.onCopy(text) }
                }
            }
            sections.addView(view)
        }
    }

    private fun shareText(result: TextReading): String = ReadTextFormat.share(
        result,
        translations.mapNotNull { (code, t) -> (t as? Translation.Ready)?.let { languageName(code, languages) to it.entries } },
    )

    private fun section(title: String, icon: Int): View {
        val view = inflater.inflate(R.layout.item_read_section, sections, false)
        view.findViewById<TextView>(R.id.section_title).text = title
        view.findViewById<ImageView>(R.id.section_icon).setImageResource(icon)
        return view
    }

    private fun fillBlocks(section: View, result: TextReading, text: (Int) -> String) {
        val box = section.findViewById<LinearLayout>(R.id.section_blocks)
        ReadBlocks.of(result.lines).forEachIndexed { i, block ->
            box.addView(
                TextView(context).apply {
                    this.text = block.lines.joinToString("\n") { text(it) }
                    setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodyLarge)
                    setTextColor(ContextCompat.getColor(context, R.color.gv_on_surface))
                    setLineSpacing(2 * dp, 1f)
                    setTextIsSelectable(true)
                },
                LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                    if (i > 0) topMargin = (14 * dp).toInt()
                },
            )
        }
    }

    private fun sectionState(section: View, busy: Boolean, error: Boolean, text: String) {
        section.findViewById<View>(R.id.section_state).visibility = View.VISIBLE
        section.findViewById<View>(R.id.section_progress).visibility = if (busy) View.VISIBLE else View.GONE
        section.findViewById<View>(R.id.section_error_icon).visibility = if (error) View.VISIBLE else View.GONE
        section.findViewById<TextView>(R.id.section_state_text).text = text
    }
}
