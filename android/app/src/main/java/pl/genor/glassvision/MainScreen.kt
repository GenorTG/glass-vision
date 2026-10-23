package pl.genor.glassvision

import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.content.res.ColorStateList
import android.graphics.drawable.Drawable
import android.view.LayoutInflater
import android.view.View
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup

/**
 * The camera screen's chrome (view_camera_chrome.xml) rendered from plain data: status pill + expandable HUD,
 * contextual action pills, one multi-select mode chip per pipeline, round controls. Knows nothing about
 * CameraX, sockets or prefs, so snapshot tests render it as is. [render] is cheap enough to call 4×/s:
 * chips and pills are only rebuilt when the set of pipelines/actions changes. Main thread only.
 */
class MainScreen(private val root: View, private val callbacks: Callbacks) {

    interface Callbacks {
        fun onTogglePipeline(id: String, enabled: Boolean)
        fun onTunePipeline(id: String)
        fun onAction(entry: ActionEntry)
        fun onSettings()
        fun onPauseToggle()
        fun onFlipCamera()
        fun onTakePhoto()
        /** Pill/banner tapped while no server is configured, the token was rejected or the server is offline. */
        fun onConnectionSetup()
    }

    data class State(
        val connection: ConnectionState,
        val connectionDetail: String?,
        val serverInfo: String?,
        val stats: StatsSnapshot,
        val paused: Boolean,
        /** Pipelines from hello; null before it / while disconnected. */
        val pipelines: List<PipelineSpec>?,
        val pipelineSettings: PipelineSettings,
        val facing: CameraFacing,
        /** Both cameras exist: show the flip button. */
        val canFlip: Boolean,
    )

    private val context = root.context
    private val inflater = LayoutInflater.from(context)

    private val statusPill: View = root.findViewById(R.id.status_pill)
    private val statusDot: View = root.findViewById(R.id.status_dot)
    private val statusText: TextView = root.findViewById(R.id.status_text)
    private val statusExpand: ImageView = root.findViewById(R.id.status_expand)
    private val hudCard: LinearLayout = root.findViewById(R.id.hud_card)
    private val actionScroll: View = root.findViewById(R.id.action_scroll)
    private val actionRow: LinearLayout = root.findViewById(R.id.action_row)
    private val chipScroll: View = root.findViewById(R.id.chip_scroll)
    private val chipGroup: ChipGroup = root.findViewById(R.id.mode_chips)
    private val banner: View = root.findViewById(R.id.chip_banner)
    private val bannerIcon: ImageView = root.findViewById(R.id.banner_icon)
    private var bannerKind: ChipBanner? = null
    private var bannerPulse: ObjectAnimator? = null
    private val bannerText: TextView = root.findViewById(R.id.banner_text)
    private val bannerDetail: TextView = root.findViewById(R.id.banner_detail)
    private val pauseButton: ImageButton = root.findViewById(R.id.pause_button)
    private val flipButton: ImageButton = root.findViewById(R.id.flip_button)

    private val chipBg: ColorStateList? = ContextCompat.getColorStateList(context, R.color.mode_chip_bg)
    private val chipStroke: ColorStateList? = ContextCompat.getColorStateList(context, R.color.mode_chip_stroke)
    private val chipText: ColorStateList? = ContextCompat.getColorStateList(context, R.color.mode_chip_text)
    private val errorBg = colors(R.color.gv_error_container)
    private val errorStroke = colors(R.color.gv_status_bad)
    private val errorText = colors(R.color.gv_on_error_container)
    private val glassBg = colors(R.color.gv_scrim_glass_strong)
    private val accentStroke = colors(R.color.gv_primary)
    private val onSurfaceText = colors(R.color.gv_on_surface)

    private val chips = LinkedHashMap<String, Chip>()
    private var actionKeys: List<String> = emptyList()
    /** Suppresses chip listeners while render() sets their checked state. */
    private var binding = false
    private var last: State? = null

    /** The detailed HUD under the status pill; toggled by tapping the pill. */
    var hudExpanded = false
        set(value) {
            field = value
            last?.let(::render)
        }

    init {
        statusPill.setOnClickListener {
            val connection = last?.connection
            if (connection != null && StatusText.needsSetup(connection)) callbacks.onConnectionSetup() else hudExpanded = !hudExpanded
        }
        banner.setOnClickListener {
            if (bannerKind in SETUP_BANNERS) callbacks.onConnectionSetup()
        }
        root.findViewById<View>(R.id.settings_button).setOnClickListener { callbacks.onSettings() }
        root.findViewById<View>(R.id.photo_button).setOnClickListener { callbacks.onTakePhoto() }
        pauseButton.setOnClickListener { callbacks.onPauseToggle() }
        flipButton.setOnClickListener { callbacks.onFlipCamera() }
    }

    fun render(state: State) {
        last = state
        renderStatus(state)
        renderChips(state)
        renderActions(state)
        renderControls(state)
    }

    private fun renderStatus(state: State) {
        val pill = StatusText.pill(state.connection, state.paused, state.stats)
        statusText.text = pill.text
        statusDot.backgroundTintList = ColorStateList.valueOf(toneColor(pill.tone))
        // Setup states: the pill opens the connection sheet, so no HUD chevron/HUD.
        val showHud = hudExpanded && !StatusText.needsSetup(state.connection)
        statusExpand.visibility = if (StatusText.needsSetup(state.connection)) View.GONE else View.VISIBLE
        statusExpand.setImageResource(if (hudExpanded) R.drawable.ic_expand_less else R.drawable.ic_expand_more)
        hudCard.visibility = if (showHud) View.VISIBLE else View.GONE
        if (!showHud) return
        val rows = StatusText.hudRows(state.connection, state.connectionDetail, state.serverInfo, state.paused, state.stats)
        while (hudCard.childCount > rows.size) hudCard.removeViewAt(hudCard.childCount - 1)
        while (hudCard.childCount < rows.size) inflater.inflate(R.layout.item_hud_row, hudCard, true)
        rows.forEachIndexed { i, row ->
            val v = hudCard.getChildAt(i)
            v.findViewById<TextView>(R.id.hud_label).text = row.label
            v.findViewById<TextView>(R.id.hud_value).text = row.value
        }
    }

    private fun renderChips(state: State) {
        val kind = StatusText.banner(state.connection, state.pipelines)
        if (kind != null) {
            chipScroll.visibility = View.GONE
            banner.visibility = View.VISIBLE
            renderBanner(kind, state.connectionDetail)
            return
        }
        if (bannerKind != null) {
            bannerKind = null
            bannerPulse?.cancel()
            bannerPulse = null
            banner.visibility = View.GONE
        }
        chipScroll.visibility = View.VISIBLE
        val models = ModeChips.of(state.pipelines.orEmpty(), state.pipelineSettings)
        if (models.map { it.id } != chips.keys.toList()) rebuildChips(models)
        binding = true
        try {
            models.forEach { m -> chips[m.id]?.let { bindChip(it, m) } }
        } finally {
            binding = false
        }
    }

    private fun renderBanner(kind: ChipBanner, detail: String?) {
        val busy = kind == ChipBanner.CONNECTING || kind == ChipBanner.WAITING_FOR_HELLO
        if (bannerKind != kind) {
            bannerKind = kind
            bannerPulse?.cancel()
            bannerPulse = null
            bannerIcon.alpha = 1f
            bannerIcon.visibility = if (kind == ChipBanner.NO_PIPELINES) View.GONE else View.VISIBLE
            banner.isClickable = kind in SETUP_BANNERS
            if (kind == ChipBanner.OFFLINE) {
                bannerIcon.setImageResource(R.drawable.ic_offline)
                bannerIcon.imageTintList = colors(R.color.gv_status_bad)
            } else if (kind == ChipBanner.NOT_CONFIGURED) {
                bannerIcon.setImageResource(R.drawable.ic_settings)
                bannerIcon.imageTintList = colors(R.color.gv_primary)
            } else if (kind == ChipBanner.UNAUTHORIZED) {
                bannerIcon.setImageResource(R.drawable.ic_error_dot)
                bannerIcon.imageTintList = null
            } else if (busy) {
                bannerIcon.setImageResource(R.drawable.ic_loading_dot)
                bannerIcon.imageTintList = null
                bannerPulse = ObjectAnimator.ofFloat(bannerIcon, View.ALPHA, 1f, 0.3f).apply {
                    duration = PULSE_MS
                    repeatMode = ValueAnimator.REVERSE
                    repeatCount = ValueAnimator.INFINITE
                    start()
                }
            }
        }
        bannerText.setText(
            when (kind) {
                ChipBanner.NOT_CONFIGURED -> R.string.banner_not_configured
                ChipBanner.UNAUTHORIZED -> R.string.banner_unauthorized
                ChipBanner.CONNECTING -> R.string.banner_connecting
                ChipBanner.OFFLINE -> R.string.banner_offline
                ChipBanner.WAITING_FOR_HELLO -> R.string.banner_waiting
                ChipBanner.NO_PIPELINES -> R.string.banner_no_pipelines
            }
        )
        val showDetail = kind == ChipBanner.OFFLINE && !detail.isNullOrBlank()
        bannerDetail.visibility = if (showDetail) View.VISIBLE else View.GONE
        if (showDetail) bannerDetail.text = detail
    }

    private fun rebuildChips(models: List<ModeChip>) {
        chips.values.forEach { (it.getTag(R.id.tag_chip_pulse) as? ValueAnimator)?.cancel() }
        chipGroup.removeAllViews()
        chips.clear()
        for (m in models) {
            val chip = inflater.inflate(R.layout.item_mode_chip, chipGroup, false) as Chip
            chip.setOnCheckedChangeListener { _, checked ->
                if (!binding) callbacks.onTogglePipeline(m.id, checked)
            }
            chip.setOnLongClickListener {
                callbacks.onTunePipeline(m.id)
                true
            }
            chip.setOnCloseIconClickListener { callbacks.onTunePipeline(m.id) }
            chipGroup.addView(chip)
            chips[m.id] = chip
        }
    }

    private fun bindChip(chip: Chip, m: ModeChip) {
        chip.text = m.label
        if (chip.isChecked != m.enabled) chip.isChecked = m.enabled
        chip.isCloseIconVisible = m.tunable
        chip.closeIconContentDescription = context.getString(R.string.chip_tune, m.label)
        chip.contentDescription =
            context.getString(if (m.enabled) R.string.chip_enabled else R.string.chip_disabled, m.label)
        // Only restyle on a state change: re-setting the icon every tick would restart its pulse.
        if (chip.getTag(R.id.tag_chip_visual) == m.visual) return
        chip.setTag(R.id.tag_chip_visual, m.visual)
        (chip.getTag(R.id.tag_chip_pulse) as? ValueAnimator)?.cancel()
        when (m.visual) {
            // On (checked) = accent fill, off = glass: both from the style's checked-state colour lists.
            ChipVisual.OFF, ChipVisual.READY -> chip.style(chipBg, chipStroke, chipText)
            // On but not ready yet: glass with an accent outline, so "ready" stays visibly different.
            ChipVisual.LOADING -> chip.style(glassBg, accentStroke, onSurfaceText)
            ChipVisual.ERROR -> chip.style(errorBg, errorStroke, errorText)
        }
        val icon: Drawable? = when (m.visual) {
            ChipVisual.OFF -> null
            ChipVisual.LOADING -> ContextCompat.getDrawable(context, R.drawable.ic_loading_dot)?.mutate()
            ChipVisual.READY -> ContextCompat.getDrawable(context, R.drawable.ic_check)?.mutate()?.apply {
                setTint(ContextCompat.getColor(context, R.color.gv_on_primary))
            }
            ChipVisual.ERROR -> ContextCompat.getDrawable(context, R.drawable.ic_error_dot)
        }
        chip.chipIcon = icon
        chip.isChipIconVisible = icon != null
        if (m.visual == ChipVisual.LOADING) chip.setTag(R.id.tag_chip_pulse, pulse(chip))
    }

    private fun Chip.style(bg: ColorStateList?, stroke: ColorStateList?, text: ColorStateList?) {
        chipBackgroundColor = bg
        chipStrokeColor = stroke
        setTextColor(text)
        closeIconTint = text
    }

    /** Fades the loading dot in and out until the chip leaves the loading state. */
    private fun pulse(chip: Chip): ValueAnimator = ValueAnimator.ofInt(255, 70).apply {
        duration = PULSE_MS
        repeatMode = ValueAnimator.REVERSE
        repeatCount = ValueAnimator.INFINITE
        addUpdateListener {
            chip.chipIcon?.alpha = it.animatedValue as Int
            chip.invalidate()
        }
        start()
    }

    private fun renderActions(state: State) {
        val entries = if (state.connection == ConnectionState.CONNECTED) {
            ActionEntries.of(state.pipelines.orEmpty(), state.pipelineSettings)
        } else {
            emptyList()
        }
        actionScroll.visibility = if (entries.isEmpty()) View.GONE else View.VISIBLE
        val keys = entries.map { it.key }
        if (keys == actionKeys) return
        actionKeys = keys
        actionRow.removeAllViews()
        for (entry in entries) {
            val pill = inflater.inflate(R.layout.item_action_pill, actionRow, false) as MaterialButton
            pill.text = entry.action.label
            ActionHints.of(entry.pipeline.id, entry.action.id).icon?.let(pill::setIconResource)
            pill.setOnClickListener { callbacks.onAction(entry) }
            actionRow.addView(pill)
        }
    }

    private fun renderControls(state: State) {
        pauseButton.setImageResource(if (state.paused) R.drawable.ic_play else R.drawable.ic_pause)
        pauseButton.contentDescription =
            context.getString(if (state.paused) R.string.resume_streaming else R.string.pause_streaming)
        // INVISIBLE, not GONE: the shutter stays centred.
        flipButton.visibility = if (state.canFlip) View.VISIBLE else View.INVISIBLE
        flipButton.contentDescription = context.getString(
            if (state.facing == CameraFacing.BACK) R.string.switch_to_front_camera else R.string.switch_to_back_camera
        )
    }

    private fun toneColor(tone: Tone): Int = ContextCompat.getColor(
        context,
        when (tone) {
            Tone.GOOD -> R.color.gv_status_good
            Tone.BUSY -> R.color.gv_status_busy
            Tone.BAD -> R.color.gv_status_bad
            Tone.IDLE -> R.color.gv_status_idle
        },
    )

    private fun colors(id: Int) = ColorStateList.valueOf(ContextCompat.getColor(context, id))

    private companion object {
        const val PULSE_MS = 700L
        /** Banners that open the connection sheet when tapped (offline too: the address may be wrong). */
        val SETUP_BANNERS = setOf(ChipBanner.NOT_CONFIGURED, ChipBanner.UNAUTHORIZED, ChipBanner.OFFLINE)
    }
}
