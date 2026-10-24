package pl.genor.glassvision

import android.content.Context
import android.graphics.Bitmap
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.core.widget.NestedScrollView
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.progressindicator.CircularProgressIndicator
import com.google.android.material.snackbar.Snackbar

/**
 * Read text results, block by block (a sign, a paragraph = one object): the still with numbered block outlines, then
 * reading… / error + retry / no text found, or one card per block — its text (prose reflowed, lists line by line), its
 * translation into every target, copy/share, phone/link/e-mail/date actions, and a "may be misread" note on suspect
 * blocks. Two views ([ReadView]): Cards (small still) and On picture (large still with each translation painted over
 * its block, "Show original" toggle, the selected target chip decides the language). Tapping a block on the still
 * highlights its card and the other way round. Plain data in, [Callbacks] out; [ReadTextFlow] does the work. Main
 * thread only.
 */
class ReadTextSheet(
    private val context: Context,
    private val callbacks: Callbacks,
    view: ReadView = ReadView.CARDS,
) {

    interface Callbacks {
        fun onRetry()
        fun onAddLanguage()
        fun onRemoveLanguage(code: String)
        fun onCopy(text: String)
        fun onShare(text: String)

        /** A data detector chip (call, open link, write e-mail, add date). */
        fun onAction(detected: Detected) = Unit

        /** The user switched views (remember it). */
        fun onViewChanged(view: ReadView) = Unit

        /** "Read carefully" on a suspect block (index into [TextReading.blocks]); only offered when [canReadCarefully]. */
        fun onReadCarefully(block: Int) = Unit
    }

    /** One target language's state. */
    sealed interface Translation {
        data object Pending : Translation
        /** Aligned with the blocks; null = untranslated (shows the original). */
        data class Ready(val entries: List<String?>) : Translation
        data class Failed(val message: String) : Translation
    }

    val sheet = Sheet(context)
    private val inflater = LayoutInflater.from(context)
    private val dp = context.resources.displayMetrics.density
    private val body = inflater.inflate(R.layout.sheet_read_text_body, null as ViewGroup?, false)
    private val still: ReadStillView = body.findViewById(R.id.read_still)
    private val viewRow: View = body.findViewById(R.id.read_view_row)
    private val viewToggle: MaterialButtonToggleGroup = body.findViewById(R.id.read_view_toggle)
    private val showOriginalChip: Chip = body.findViewById(R.id.read_show_original)
    private val targetsScroll: View = body.findViewById(R.id.read_targets_scroll)
    private val targetChips: ChipGroup = body.findViewById(R.id.read_targets)
    private val targetStatus: TextView = body.findViewById(R.id.read_target_status)
    private val status: View = body.findViewById(R.id.read_status)
    private val progress: CircularProgressIndicator = body.findViewById(R.id.read_progress)
    private val statusIcon: ImageView = body.findViewById(R.id.read_status_icon)
    private val statusText: TextView = body.findViewById(R.id.read_status_text)
    private val statusDetail: TextView = body.findViewById(R.id.read_status_detail)
    private val retry: MaterialButton = body.findViewById(R.id.read_retry)
    private val blockList: LinearLayout = body.findViewById(R.id.read_blocks)
    private val allActions: View = body.findViewById(R.id.read_all_actions)

    private var highRes = true
    private var reading: TextReading? = null
    private val translations = LinkedHashMap<String, Translation>()
    private val cards = ArrayList<View>()

    /** The language painted on the picture in [ReadView.OVERLAY] (null = the first ready target). */
    private var overlayTarget: String? = null
    private var showOriginal = false

    /** Index of the highlighted block, or -1. */
    var selected: Int = -1
        private set

    var view: ReadView = view
        private set

    /** Names for language codes ([TextReader.languages]); codes without one use the phone's name. */
    var languages: List<TranslationLanguage> = emptyList()
        set(value) {
            field = value
            if (reading != null) renderResult()
        }

    /** The server offers `POST /text/careful`: suspect blocks get an enabled "Read carefully". */
    var canReadCarefully: Boolean = false
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
        body.findViewById<View>(R.id.read_copy_all).setOnClickListener { reading?.let { callbacks.onCopy(shareText(it)) } }
        body.findViewById<View>(R.id.read_share_all).setOnClickListener { reading?.let { callbacks.onShare(shareText(it)) } }
        viewToggle.check(if (view == ReadView.OVERLAY) R.id.read_view_overlay else R.id.read_view_cards)
        viewToggle.addOnButtonCheckedListener { _, id, checked ->
            if (!checked) return@addOnButtonCheckedListener
            val chosen = if (id == R.id.read_view_overlay) ReadView.OVERLAY else ReadView.CARDS
            if (chosen == this.view) return@addOnButtonCheckedListener
            this.view = chosen
            callbacks.onViewChanged(chosen)
            if (reading != null) renderResult()
        }
        showOriginalChip.setOnCheckedChangeListener { _, checked ->
            showOriginal = checked
            applyStill()
        }
        still.onBlockTap = { select(it, fromStill = true) }
    }

    /** The picture being read (a downscaled copy is enough); [highRes] = false for the analysis-frame fallback. */
    fun setStill(picture: Bitmap?, highRes: Boolean) {
        this.highRes = highRes
        still.setImageBitmap(picture)
        still.visibility = if (picture != null) View.VISIBLE else View.GONE
    }

    fun showReading() {
        reading = null
        translations.clear()
        still.blocks = emptyList()
        still.overlay = null
        still.fitToImage = false
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
        selected = -1
        translations.clear()
        val done = result.translated
        for (code in (targets + done + result.errors.keys).distinct()) {
            translations[code] = if (code in done) {
                Translation.Ready(result.translation(code))
            } else {
                Translation.Failed(result.errors[code] ?: "not translated")
            }
        }
        sheet.setSubtitle(subtitle(result))
        if (result.blocks.isEmpty()) {
            still.blocks = emptyList()
            still.overlay = null
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
        if (translations.remove(code) != null && reading != null) {
            if (overlayTarget == code) overlayTarget = null
            renderResult()
        }
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

    /** Highlights block [index] on the still and its card; [fromStill] = tapped on the picture (scroll to the card). */
    fun select(index: Int, fromStill: Boolean = false) {
        val result = reading ?: return
        if (index !in result.blocks.indices) return
        selected = index
        still.selected = index
        cards.forEachIndexed { i, card -> card.setBackgroundResource(if (i == index) R.drawable.read_block_selected_bg else R.drawable.read_block_bg) }
        val scroll = sheet.view.findViewById<NestedScrollView>(R.id.sheet_scroll) ?: return
        val target = if (fromStill) cards.getOrNull(index) else still
        target?.let { v -> scroll.post { scrollToShow(scroll, v) } }
    }

    private fun scrollToShow(scroll: NestedScrollView, v: View) {
        var top = 0
        var p: View? = v
        while (p != null && p !== scroll) {
            top += p.top
            p = p.parent as? View
        }
        val visibleTop = scroll.scrollY
        val visibleBottom = visibleTop + scroll.height
        if (top < visibleTop || top + v.height > visibleBottom) scroll.smoothScrollTo(0, (top - (8 * dp).toInt()).coerceAtLeast(0))
    }

    private fun subtitle(result: TextReading): String {
        val parts = mutableListOf<String>()
        if (result.blocks.isNotEmpty()) {
            parts += context.resources.getQuantityString(R.plurals.read_blocks, result.blocks.size, result.blocks.size)
        }
        result.script?.takeIf { it.isNotBlank() }?.let { parts += it.replaceFirstChar(Char::titlecase) }
        if (!highRes) parts += context.getString(R.string.read_low_res)
        return parts.joinToString(" · ")
    }

    private fun showStatus(busy: Boolean, icon: Int?, title: Int, detail: String?, retry: Boolean = false) {
        status.visibility = View.VISIBLE
        viewRow.visibility = View.GONE
        showOriginalChip.visibility = View.GONE
        targetsScroll.visibility = View.GONE
        targetStatus.visibility = View.GONE
        blockList.visibility = View.GONE
        allActions.visibility = View.GONE
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

    /** The target painted on the picture: the chosen one if it is ready, else the first ready one. */
    private fun overlayCode(): String? {
        val ready = translations.filterValues { it is Translation.Ready }.keys
        return overlayTarget?.takeIf { it in ready } ?: ready.firstOrNull()
    }

    private fun applyStill() {
        val result = reading ?: return
        still.blocks = result.blocks
        still.selected = selected
        val overlay = view == ReadView.OVERLAY
        still.fitToImage = overlay
        val code = overlayCode()
        still.overlay = if (overlay && !showOriginal && code != null) (translations[code] as Translation.Ready).entries else null
        still.contentDescription = context.getString(if (still.overlay != null) R.string.read_still_overlay else R.string.read_thumbnail)
    }

    private fun renderResult() {
        val result = reading ?: return
        if (result.blocks.isEmpty()) return
        status.visibility = View.GONE
        viewRow.visibility = View.VISIBLE
        showOriginalChip.visibility = if (view == ReadView.OVERLAY) View.VISIBLE else View.GONE
        applyStill()
        renderTargets()
        renderCards(result)
        allActions.visibility = View.VISIBLE
    }

    private fun renderTargets() {
        targetsScroll.visibility = View.VISIBLE
        targetChips.removeAllViews()
        val overlay = view == ReadView.OVERLAY
        val shown = overlayCode()
        val notes = ArrayList<String>()
        for ((code, translation) in translations) {
            val name = languageName(code, languages)
            val chip = inflater.inflate(R.layout.item_read_language_chip, targetChips, false) as Chip
            chip.text = if (translation is Translation.Pending) context.getString(R.string.read_target_pending, name) else name
            chip.closeIconContentDescription = context.getString(R.string.read_remove_language, name)
            chip.setOnCloseIconClickListener { callbacks.onRemoveLanguage(code) }
            chip.isCheckable = overlay && translation is Translation.Ready
            chip.isChecked = overlay && code == shown
            if (translation is Translation.Failed) {
                chip.chipIcon = ContextCompat.getDrawable(context, R.drawable.ic_error_dot)
                chip.isChipIconVisible = true
            }
            chip.setOnClickListener {
                if (translation is Translation.Ready && overlay) {
                    overlayTarget = code
                    renderTargets()
                    applyStill()
                }
            }
            targetChips.addView(chip)
            when (translation) {
                Translation.Pending -> notes += context.getString(R.string.read_translating_into, name)
                is Translation.Failed -> notes += context.getString(R.string.read_translation_failed_into, name, translation.message)
                is Translation.Ready -> if (ReadTextFormat.isUnchanged(translation.entries)) notes += context.getString(R.string.read_unchanged, name)
            }
        }
        val add = inflater.inflate(R.layout.item_read_assist_chip, targetChips, false) as Chip
        add.setText(R.string.read_add_language)
        add.chipIcon = ContextCompat.getDrawable(context, R.drawable.ic_add)
        add.setOnClickListener { callbacks.onAddLanguage() }
        targetChips.addView(add)
        if (overlay && shown == null) notes += context.getString(R.string.read_overlay_no_target)
        targetStatus.text = notes.joinToString("\n")
        targetStatus.visibility = if (notes.isEmpty()) View.GONE else View.VISIBLE
    }

    private fun renderCards(result: TextReading) {
        blockList.visibility = View.VISIBLE
        blockList.removeAllViews()
        cards.clear()
        val ready = translations.mapNotNull { (code, t) -> (t as? Translation.Ready)?.let { code to it.entries } }
        for ((i, block) in result.blocks.withIndex()) {
            val card = inflater.inflate(R.layout.item_read_block, blockList, false)
            card.findViewById<TextView>(R.id.block_number).apply {
                text = (i + 1).toString()
                if (block.suspect) {
                    setBackgroundResource(R.drawable.read_badge_suspect_bg)
                    setTextColor(ContextCompat.getColor(context, R.color.gv_on_status_busy))
                }
            }
            val language = block.lang?.let { languageName(it, languages) } ?: context.getString(R.string.read_unknown_language)
            card.findViewById<TextView>(R.id.block_label).text =
                context.getString(if (block.kind == BlockKind.PROSE) R.string.read_kind_prose else R.string.read_kind_list, language)
            card.findViewById<TextView>(R.id.block_original).text = block.display

            val box = card.findViewById<LinearLayout>(R.id.block_translations)
            val named = ArrayList<Pair<String, String?>>()
            for ((code, entries) in ready) {
                val t = entries.getOrNull(i) ?: continue
                val name = languageName(code, languages)
                named += name to t
                box.addView(
                    TextView(context).apply {
                        text = name
                        setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_LabelMedium)
                        setTextColor(ContextCompat.getColor(context, R.color.gv_primary))
                    },
                    LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = (10 * dp).toInt() },
                )
                box.addView(
                    TextView(context).apply {
                        text = t
                        setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodyLarge)
                        setTextColor(ContextCompat.getColor(context, R.color.gv_on_surface))
                        setLineSpacing(2 * dp, 1f)
                        setTextIsSelectable(true)
                    },
                )
            }

            card.findViewById<View>(R.id.block_suspect).visibility = if (block.suspect) View.VISIBLE else View.GONE
            card.findViewById<MaterialButton>(R.id.block_careful).apply {
                isEnabled = canReadCarefully
                setOnClickListener { callbacks.onReadCarefully(i) }
            }

            val actions = card.findViewById<ChipGroup>(R.id.block_actions)
            for (d in TextDetectors.entities(block.display)) {
                val chip = inflater.inflate(R.layout.item_read_assist_chip, actions, false) as Chip
                chip.text = d.text
                chip.chipIcon = ContextCompat.getDrawable(context, ACTION_ICONS.getValue(d.kind))
                chip.setOnClickListener { callbacks.onAction(d) }
                actions.addView(chip)
            }
            actions.visibility = if (actions.childCount > 0) View.VISIBLE else View.GONE

            val text = ReadTextFormat.block(block, named)
            card.findViewById<View>(R.id.block_copy).setOnClickListener { callbacks.onCopy(text) }
            card.findViewById<View>(R.id.block_share).setOnClickListener { callbacks.onShare(text) }
            card.setOnClickListener { select(i) }
            if (i == selected) card.setBackgroundResource(R.drawable.read_block_selected_bg)
            blockList.addView(card)
            cards += card
        }
    }

    private fun shareText(result: TextReading): String = ReadTextFormat.share(
        result,
        translations.mapNotNull { (code, t) -> (t as? Translation.Ready)?.let { languageName(code, languages) to it.entries } },
    )

    private companion object {
        val ACTION_ICONS = mapOf(
            DetectedKind.PHONE to R.drawable.ic_call,
            DetectedKind.URL to R.drawable.ic_link,
            DetectedKind.EMAIL to R.drawable.ic_mail,
            DetectedKind.DATE to R.drawable.ic_event,
        )
    }
}
