package pl.genor.glassvision

import android.content.Context

/**
 * One pipeline's options and actions in a bottom sheet (long-press or ⚙ on its mode chip): the same
 * [PipelineCard] as in Settings, kept live by [catalog] status changes, closed when the server's pipeline set
 * changes (reconnect). Main thread only.
 */
class PipelineSheet(
    context: Context,
    spec: PipelineSpec,
    private val catalog: PipelineCatalog,
    choices: PipelineChoices,
    onAction: (PipelineSpec, ActionSpec) -> Unit,
) {
    val sheet = Sheet(context)
    private val card = PipelineCard(context, spec, choices, onAction)

    private val onCatalogChange: (PipelineCatalog.Change) -> Unit = { change ->
        when (change) {
            PipelineCatalog.Change.Rebuilt -> sheet.dismiss()
            is PipelineCatalog.Change.Status -> if (change.pipeline == card.id) catalog.find(card.id)?.let(card::update)
        }
    }

    init {
        // No sheet title: the card's own header (name, status, switch) is the heading.
        sheet.setBody(card.view)
    }

    fun show() {
        catalog.addListener(onCatalogChange)
        sheet.onDismiss {
            catalog.removeListener(onCatalogChange)
            card.commit()
        }
        sheet.show()
    }
}
