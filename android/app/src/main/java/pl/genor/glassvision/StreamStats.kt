package pl.genor.glassvision

import java.util.ArrayDeque
import java.util.Locale
import kotlin.math.roundToInt

data class StatsSnapshot(
    val sentFps: Float,
    val resultFps: Float,
    val avgRttMs: Float?,
    /** Server time for the last result (decode + inline pipelines + tracking). */
    val serverMs: Double?,
    /** Latest inference ms per pipeline, in first-seen order (slow pipelines don't run every frame). */
    val pipelineMs: Map<String, Double>,
    val dropCount: Int,
    val errorCount: Int,
    val objectCount: Int,
)

/** Thread-safe HUD counters; written from the analysis + socket threads, read on the main thread. */
class StreamStats(private val clock: () -> Long = System::currentTimeMillis) {
    private val sentTimes = ArrayDeque<Long>()
    private val resultTimes = ArrayDeque<Long>()
    private val rtts = ArrayDeque<Long>()
    private var serverMs: Double? = null
    /** pipeline → (ms, when reported). */
    private val pipelineMs = LinkedHashMap<String, Pair<Double, Long>>()
    private var drops = 0
    private var errors = 0
    private var objects = 0

    @Synchronized
    fun onSent() = sentTimes.addLast(clock())

    @Synchronized
    fun onDetections(rttMs: Long, serverMs: Double, pipelineMs: Map<String, Double>, objectCount: Int) {
        val now = clock()
        resultTimes.addLast(now)
        rtts.addLast(rttMs)
        while (rtts.size > RTT_SAMPLES) rtts.removeFirst()
        this.serverMs = serverMs
        pipelineMs.forEach { (id, ms) -> this.pipelineMs[id] = ms to now }
        objects = objectCount
    }

    @Synchronized
    fun onDropped() {
        drops++
    }

    @Synchronized
    fun onError() {
        errors++
    }

    @Synchronized
    fun reset() {
        sentTimes.clear()
        resultTimes.clear()
        rtts.clear()
        serverMs = null
        pipelineMs.clear()
        drops = 0
        errors = 0
        objects = 0
    }

    @Synchronized
    fun snapshot(): StatsSnapshot {
        val now = clock()
        prune(sentTimes, now)
        prune(resultTimes, now)
        // A pipeline that stopped reporting (disabled) drops off the HUD.
        pipelineMs.entries.removeAll { now - it.value.second > PIPELINE_MS_TTL }
        val seconds = RATE_WINDOW_MS / 1000f
        return StatsSnapshot(
            sentFps = sentTimes.size / seconds,
            resultFps = resultTimes.size / seconds,
            avgRttMs = if (rtts.isEmpty()) null else rtts.average().toFloat(),
            serverMs = serverMs,
            pipelineMs = pipelineMs.mapValuesTo(LinkedHashMap()) { it.value.first },
            dropCount = drops,
            errorCount = errors,
            objectCount = objects,
        )
    }

    private fun prune(times: ArrayDeque<Long>, now: Long) {
        while (times.isNotEmpty() && now - times.first() > RATE_WINDOW_MS) times.removeFirst()
    }

    companion object {
        private const val RATE_WINDOW_MS = 2000L
        private const val RTT_SAMPLES = 20
        private const val PIPELINE_MS_TTL = 5000L

        /** "objects 8 · faces 4 · text 120" (ms; one decimal below 10). */
        fun formatPipelineMs(ms: Map<String, Double>): String =
            ms.entries.joinToString(" · ") { (id, v) ->
                id + " " + if (v < 10) String.format(Locale.US, "%.1f", v) else v.roundToInt().toString()
            }
    }
}
