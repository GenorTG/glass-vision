package pl.genor.glassvision

/**
 * The pipelines the connected server offers (from `hello`, statuses kept current by `pipeline_status`).
 * [specs] is null while there is no connection/hello. Main thread only.
 */
class PipelineCatalog {
    sealed interface Change {
        /** New hello or disconnect: rebuild anything derived from [specs]. */
        data object Rebuilt : Change

        data class Status(val pipeline: String) : Change
    }

    var specs: List<PipelineSpec>? = null
        private set
    private val listeners = mutableListOf<(Change) -> Unit>()

    fun find(id: String): PipelineSpec? = specs?.firstOrNull { it.id == id }

    fun onHello(hello: ServerMessage.Hello) {
        specs = hello.pipelines
        notify(Change.Rebuilt)
    }

    fun onStatus(change: ServerMessage.PipelineStatusChange) {
        val current = specs ?: return
        if (current.none { it.id == change.pipeline }) return
        specs = current.map {
            if (it.id == change.pipeline) it.copy(status = change.status, statusMessage = change.message) else it
        }
        notify(Change.Status(change.pipeline))
    }

    fun clear() {
        if (specs == null) return
        specs = null
        notify(Change.Rebuilt)
    }

    fun addListener(l: (Change) -> Unit) {
        listeners += l
    }

    fun removeListener(l: (Change) -> Unit) {
        listeners -= l
    }

    private fun notify(change: Change) = listeners.toList().forEach { it(change) }
}
