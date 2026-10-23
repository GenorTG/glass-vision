package pl.genor.glassvision

import android.content.Context
import android.content.res.ColorStateList
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.materialswitch.MaterialSwitch

/**
 * One pipeline as a card, built from its hello description alone: name + live status + on/off switch,
 * description, typed option editors (saved to [choices] → FrameStreamer sends the delta) and action buttons.
 * Used by Settings → Recognition and by the per-pipeline sheet (long-press on a mode chip). Main thread only.
 */
class PipelineCard(
    private val context: Context,
    spec: PipelineSpec,
    private val choices: PipelineChoices,
    private val onAction: (PipelineSpec, ActionSpec) -> Unit,
) {
    val id: String = spec.id
    val view: View = LayoutInflater.from(context).inflate(R.layout.item_pipeline_card, null as ViewGroup?, false)
    private val statusDot: View = view.findViewById(R.id.card_status_dot)
    private val status: TextView = view.findViewById(R.id.card_status)
    private val switch: MaterialSwitch = view.findViewById(R.id.card_switch)
    private val editors = mutableListOf<OptionEditor>()
    private var spec: PipelineSpec = spec

    init {
        view.layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            .apply { bottomMargin = (10 * context.resources.displayMetrics.density).toInt() }
        view.findViewById<TextView>(R.id.card_title).text = spec.name
        val about = listOfNotNull(
            spec.description.takeIf { it.isNotBlank() },
            if (spec.intervalMs > 0) context.getString(R.string.recognition_interval, spec.intervalMs) else null,
        ).joinToString("\n")
        view.findViewById<TextView>(R.id.card_description).apply {
            text = about
            visibility = if (about.isEmpty()) View.GONE else View.VISIBLE
        }

        val settings = choices.load()
        switch.isChecked = settings.isEnabled(spec)
        switch.contentDescription = spec.name
        switch.setOnCheckedChangeListener { _, checked ->
            choices.update { it.withEnabled(spec.id, checked) }
            renderStatus()
        }
        view.findViewById<View>(R.id.card_header).setOnClickListener { switch.toggle() }

        val optionBox = view.findViewById<LinearLayout>(R.id.card_options)
        val factory = OptionEditors(context)
        for (option in spec.options) {
            val editor = factory.create(option, settings.optionValue(spec, option)) { v ->
                choices.update { it.withOption(spec.id, option.key, v) }
            }
            editors += editor
            optionBox.addView(editor.view)
        }

        if (spec.actions.isNotEmpty()) {
            view.findViewById<View>(R.id.card_actions_scroll).visibility = View.VISIBLE
            val row = view.findViewById<LinearLayout>(R.id.card_actions)
            val inflater = LayoutInflater.from(context)
            for (action in spec.actions) {
                val button = inflater.inflate(R.layout.item_card_action, row, false) as MaterialButton
                button.text = action.label
                ActionHints.of(spec.id, action.id).icon?.let(button::setIconResource)
                button.setOnClickListener { onAction(this.spec, action) }
                row.addView(button)
            }
        }
        renderStatus()
    }

    /** New status from pipeline_status (same pipeline, same options). */
    fun update(spec: PipelineSpec) {
        this.spec = spec
        renderStatus()
    }

    /** Applies text typed but not yet confirmed (sheet closing). */
    fun commit() = editors.forEach(OptionEditor::commit)

    private fun renderStatus() {
        val visual = ModeChips.visualOf(spec.status, switch.isChecked)
        val (text, color) = when (visual) {
            ChipVisual.OFF -> context.getString(R.string.recognition_status_off) to R.color.gv_status_idle
            ChipVisual.LOADING -> context.getString(R.string.recognition_status_loading) to R.color.gv_status_busy
            ChipVisual.READY -> context.getString(R.string.recognition_status_ready) to R.color.gv_status_good
            ChipVisual.ERROR -> (
                spec.statusMessage?.let { context.getString(R.string.recognition_status_error, it) }
                    ?: context.getString(R.string.recognition_status_error_plain)
                ) to R.color.gv_status_bad
        }
        val c = ContextCompat.getColor(context, color)
        status.text = text
        status.setTextColor(c)
        statusDot.backgroundTintList = ColorStateList.valueOf(c)
    }
}

/**
 * All pipelines of [catalog] as [PipelineCard]s in [container]; rebuilds on a new hello, updates statuses in
 * place. [attach] while visible, [detach] when the sheet closes (also commits pending text).
 */
class RecognitionSection(
    private val context: Context,
    private val container: LinearLayout,
    private val catalog: PipelineCatalog,
    private val choices: PipelineChoices,
    private val onAction: (PipelineSpec, ActionSpec) -> Unit,
) {
    private val cards = LinkedHashMap<String, PipelineCard>()

    private val onCatalogChange: (PipelineCatalog.Change) -> Unit = { change ->
        when (change) {
            PipelineCatalog.Change.Rebuilt -> rebuild()
            is PipelineCatalog.Change.Status -> catalog.find(change.pipeline)?.let { cards[it.id]?.update(it) }
        }
    }

    fun attach() {
        rebuild()
        catalog.addListener(onCatalogChange)
    }

    fun detach() {
        catalog.removeListener(onCatalogChange)
        cards.values.forEach(PipelineCard::commit)
    }

    /** After "Defaults": rebuild from the reset choices, dropping unapplied text. */
    fun rebuild() {
        container.removeAllViews()
        cards.clear()
        val specs = catalog.specs
        if (specs.isNullOrEmpty()) {
            container.addView(
                TextView(context).apply {
                    setText(if (specs == null) R.string.recognition_not_connected else R.string.recognition_empty)
                    setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodyMedium)
                    setTextColor(ContextCompat.getColor(context, R.color.gv_on_surface_variant))
                    val pad = (16 * context.resources.displayMetrics.density).toInt()
                    setPadding(pad, pad / 2, pad, pad)
                }
            )
            return
        }
        for (spec in specs) {
            val card = PipelineCard(context, spec, choices, onAction)
            cards[spec.id] = card
            container.addView(card.view)
        }
    }
}
