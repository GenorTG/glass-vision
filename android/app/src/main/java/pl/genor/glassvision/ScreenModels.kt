package pl.genor.glassvision

import java.util.Locale

/*
 * What the main screen shows, derived from plain data (hello, user choices, connection, stats). Pure and
 * JVM-testable; MainScreen only turns these into views.
 */

/** Chip label from a pipeline name: whole if it fits, else without its "(…)" detail, else ellipsized. */
object PipelineLabels {
    const val MAX_CHIP_CHARS = 14

    fun short(name: String): String {
        val full = name.trim()
        if (full.length <= MAX_CHIP_CHARS) return full
        val base = full.substringBefore(" (").trim().ifEmpty { full }
        return if (base.length <= MAX_CHIP_CHARS) base else base.take(MAX_CHIP_CHARS - 1).trimEnd() + "…"
    }
}

enum class ChipVisual { OFF, LOADING, READY, ERROR }

/** One mode chip: tap toggles [enabled]; [tunable] = has options/actions to open in a sheet. */
data class ModeChip(
    val id: String,
    val label: String,
    val enabled: Boolean,
    val visual: ChipVisual,
    val tunable: Boolean,
    val errorMessage: String?,
)

object ModeChips {
    fun of(specs: List<PipelineSpec>, settings: PipelineSettings): List<ModeChip> = specs.map { spec ->
        val enabled = settings.isEnabled(spec)
        ModeChip(
            id = spec.id,
            label = PipelineLabels.short(spec.name),
            enabled = enabled,
            visual = visualOf(spec.status, enabled),
            tunable = spec.options.isNotEmpty() || spec.actions.isNotEmpty(),
            errorMessage = spec.statusMessage.takeIf { spec.status == PipelineStatus.ERROR },
        )
    }

    /**
     * Errors show whether on or off (the user needs to see why it doesn't work); an enabled pipeline that
     * is not loaded yet is about to load. Servers without statuses (UNKNOWN) count as ready.
     */
    fun visualOf(status: PipelineStatus, enabled: Boolean): ChipVisual = when {
        status == PipelineStatus.ERROR -> ChipVisual.ERROR
        !enabled -> ChipVisual.OFF
        status == PipelineStatus.LOADING || status == PipelineStatus.NOT_LOADED -> ChipVisual.LOADING
        else -> ChipVisual.READY
    }
}

/** A pill in the contextual action row: [action] of the enabled [pipeline]. */
data class ActionEntry(val pipeline: PipelineSpec, val action: ActionSpec) {
    val key: String get() = "${pipeline.id}/${action.id}"
}

object ActionEntries {
    /** Actions of the enabled pipelines, in hello order. */
    fun of(specs: List<PipelineSpec>, settings: PipelineSettings): List<ActionEntry> =
        specs.filter(settings::isEnabled).flatMap { spec -> spec.actions.map { ActionEntry(spec, it) } }
}

enum class Tone { GOOD, BUSY, BAD, IDLE }

data class StatusPillModel(val text: String, val tone: Tone)

/**
 * Shown instead of the chip row while there is nothing to choose from. [STARTING_LOCAL] = the on-device engine
 * is not running yet (it has no connection to lose).
 */
enum class ChipBanner { NOT_CONFIGURED, UNAUTHORIZED, CONNECTING, OFFLINE, WAITING_FOR_HELLO, NO_PIPELINES, STARTING_LOCAL }

data class HudRow(val label: String, val value: String)

object StatusText {
    /** States where tapping the pill/banner opens the connection sheet instead of the HUD (never on the phone). */
    fun needsSetup(connection: ConnectionState, mode: ProcessingMode = ProcessingMode.SERVER): Boolean =
        mode == ProcessingMode.SERVER &&
            (connection == ConnectionState.NOT_CONFIGURED || connection == ConnectionState.UNAUTHORIZED)

    fun pill(
        connection: ConnectionState,
        paused: Boolean,
        stats: StatsSnapshot,
        mode: ProcessingMode = ProcessingMode.SERVER,
    ): StatusPillModel = when {
        mode == ProcessingMode.LOCAL -> when {
            connection != ConnectionState.CONNECTED -> StatusPillModel("Starting…", Tone.BUSY)
            paused -> StatusPillModel("Paused", Tone.IDLE)
            else -> StatusPillModel("On device · ${fmt("%.1f", stats.resultFps)} fps", Tone.GOOD)
        }
        connection == ConnectionState.NOT_CONFIGURED -> StatusPillModel("Not configured", Tone.IDLE)
        connection == ConnectionState.UNAUTHORIZED -> StatusPillModel("Wrong access token", Tone.BAD)
        connection == ConnectionState.CONNECTING -> StatusPillModel("Connecting…", Tone.BUSY)
        connection == ConnectionState.DISCONNECTED -> StatusPillModel("Offline", Tone.BAD)
        paused -> StatusPillModel("Paused", Tone.IDLE)
        else -> StatusPillModel(
            "Live · ${fmt("%.1f", stats.resultFps)} fps · ${stats.avgRttMs?.let { fmt("%.0f", it) } ?: "–"} ms",
            Tone.GOOD,
        )
    }

    /** "v0.4.0.3 · RTX 3090" from hello's version + "cuda:0 (NVIDIA GeForce RTX 3090)": fits the HUD row. */
    fun serverInfo(version: String, device: String): String {
        val gpu = device.substringAfter('(', "").substringBeforeLast(')', "").ifBlank { device }
            .removePrefix("NVIDIA ").removePrefix("GeForce ").trim()
        return "v${version.ifEmpty { "?" }}" + if (gpu.isNotEmpty()) " · $gpu" else ""
    }

    /** Null = show the chips. */
    fun banner(
        connection: ConnectionState,
        specs: List<PipelineSpec>?,
        mode: ProcessingMode = ProcessingMode.SERVER,
    ): ChipBanner? = when {
        mode == ProcessingMode.LOCAL && (connection != ConnectionState.CONNECTED || specs == null) -> ChipBanner.STARTING_LOCAL
        connection == ConnectionState.NOT_CONFIGURED -> ChipBanner.NOT_CONFIGURED
        connection == ConnectionState.UNAUTHORIZED -> ChipBanner.UNAUTHORIZED
        connection == ConnectionState.CONNECTING -> ChipBanner.CONNECTING
        connection == ConnectionState.DISCONNECTED -> ChipBanner.OFFLINE
        specs == null -> ChipBanner.WAITING_FOR_HELLO
        specs.isEmpty() -> ChipBanner.NO_PIPELINES
        else -> null
    }

    /**
     * The expanded HUD: everything the old text HUD showed, one fact per row. On the phone: engine instead of
     * server, latency = camera frame → result, engine time = all pipelines of one frame.
     */
    fun hudRows(
        connection: ConnectionState,
        detail: String?,
        serverInfo: String?,
        paused: Boolean,
        stats: StatsSnapshot,
        mode: ProcessingMode = ProcessingMode.SERVER,
    ): List<HudRow> = if (mode == ProcessingMode.LOCAL) localHudRows(connection, serverInfo, paused, stats) else buildList {
        val state = when (connection) {
            ConnectionState.CONNECTED -> if (paused) "connected · paused" else "connected"
            ConnectionState.CONNECTING -> "connecting"
            ConnectionState.DISCONNECTED -> "disconnected"
            ConnectionState.NOT_CONFIGURED -> "no server configured"
            ConnectionState.UNAUTHORIZED -> "access token rejected"
        }
        add(HudRow("State", state))
        if (connection != ConnectionState.CONNECTED) detail?.let { add(HudRow("Detail", it)) }
        serverInfo?.let { add(HudRow("Server", it)) }
        add(HudRow("Send / results", "${fmt("%.1f", stats.sentFps)} / ${fmt("%.1f", stats.resultFps)} fps"))
        add(HudRow("Round trip", stats.avgRttMs?.let { fmt("%.0f ms", it) } ?: "–"))
        add(HudRow("Server time", stats.serverMs?.let { fmt("%.1f ms", it) } ?: "–"))
        for ((id, ms) in stats.pipelineMs) add(HudRow("  $id", StatsSnapshotFormat.ms(ms)))
        add(HudRow("Drops / errors", "${stats.dropCount} / ${stats.errorCount}"))
        add(HudRow("Tracks", stats.objectCount.toString()))
    }

    private fun localHudRows(
        connection: ConnectionState,
        engineInfo: String?,
        paused: Boolean,
        stats: StatsSnapshot,
    ): List<HudRow> = buildList {
        val state = when {
            connection != ConnectionState.CONNECTED -> "starting"
            paused -> "on this phone · paused"
            else -> "on this phone"
        }
        add(HudRow("State", state))
        engineInfo?.let { add(HudRow("Engine", it)) }
        add(HudRow("Frames / results", "${fmt("%.1f", stats.sentFps)} / ${fmt("%.1f", stats.resultFps)} fps"))
        add(HudRow("Latency", stats.avgRttMs?.let { fmt("%.0f ms", it) } ?: "–"))
        add(HudRow("Engine time", stats.serverMs?.let { fmt("%.1f ms", it) } ?: "–"))
        for ((id, ms) in stats.pipelineMs) add(HudRow("  $id", StatsSnapshotFormat.ms(ms)))
        add(HudRow("Tracks", stats.objectCount.toString()))
    }

    private fun fmt(pattern: String, vararg args: Any): String = String.format(Locale.US, pattern, *args)
}

object StatsSnapshotFormat {
    /** "8.1 ms" below 10, "55 ms" above (same precision as [StreamStats.formatPipelineMs]). */
    fun ms(v: Double): String =
        if (v < 10) String.format(Locale.US, "%.1f ms", v) else String.format(Locale.US, "%.0f ms", v)
}
