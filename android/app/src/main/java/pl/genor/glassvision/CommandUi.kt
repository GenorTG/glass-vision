package pl.genor.glassvision

import android.content.Context
import com.google.android.material.dialog.MaterialAlertDialogBuilder

/**
 * Pipeline actions end to end: actions without args are sent at once; with args an [ActionSheet] asks for
 * them (pre-filled with the last values used, [ArgMemory]). Each `command_result` goes back where it came
 * from: into the open action sheet, into an [ItemListSheet] when it carries items (refreshed after an item
 * action), else onto the main screen via [MessageHost]. Main thread only.
 */
class CommandUi(
    private val context: Context,
    private val catalog: PipelineCatalog,
    private val sendCommand: (pipeline: String, action: String, args: Map<String, OptionValue>) -> Long?,
    private val messages: MessageHost,
    private val argMemory: ArgMemory,
    private val camera: CameraHost,
) {
    fun interface MessageHost {
        fun showMessage(text: String, error: Boolean)
    }

    interface CameraHost {
        val facing: CameraFacing
        val availableCameras: Set<CameraFacing>

        fun switchCamera(facing: CameraFacing)
    }

    private sealed interface Origin {
        data object Main : Origin
        class FromActionSheet(val sheet: ActionSheet) : Origin
        class FromList(val sheet: ItemListSheet) : Origin
    }

    private class Pending(val pipeline: String, val origin: Origin)

    private val pending = HashMap<Long, Pending>()
    private var actionSheet: ActionSheet? = null
    private var listSheet: ItemListSheet? = null

    fun run(pipeline: PipelineSpec, action: ActionSpec) {
        if (action.args.isEmpty()) {
            send(pipeline.id, action.id, emptyMap(), Origin.Main)
            return
        }
        actionSheet?.sheet?.dismiss()
        lateinit var sheet: ActionSheet
        sheet = ActionSheet(
            context, pipeline, action,
            initialArgs = argMemory.load(pipeline.id, action),
            facing = camera.facing,
            frontAvailable = CameraFacing.FRONT in camera.availableCameras,
            onSwitchToFront = { camera.switchCamera(CameraFacing.FRONT) },
        ) { args ->
            argMemory.save(pipeline.id, action.id, args)
            val sent = send(pipeline.id, action.id, args, if (sheet.presentation.keepOpen) Origin.FromActionSheet(sheet) else Origin.Main)
            when {
                !sent -> sheet.showResult(false, context.getString(R.string.command_not_connected))
                sheet.presentation.keepOpen -> sheet.showSending()
                else -> sheet.sheet.dismiss()
            }
        }
        sheet.sheet.onDismiss { if (actionSheet === sheet) actionSheet = null }
        actionSheet = sheet
        // Light dim: capture flows need the camera (and its face box) visible above the sheet.
        sheet.sheet.show(dimAmount = if (sheet.presentation.keepOpen) CAPTURE_DIM else null)
    }

    /** The camera switched (flip button or the sheet's offer). */
    fun onCameraChanged(facing: CameraFacing) {
        actionSheet?.onCameraChanged(facing)
    }

    fun onResult(result: ServerMessage.CommandResult) {
        val p = pending.remove(result.req)
        val pipeline = result.pipeline.ifEmpty { p?.pipeline.orEmpty() }
        val message = result.message.ifEmpty {
            context.getString(if (result.ok) R.string.command_ok else R.string.command_failed, result.action)
        }
        val origin = p?.origin
        when {
            result.items.isNotEmpty() || result.itemAction != null -> showList(pipeline, result)
            origin is Origin.FromActionSheet && origin.sheet.sheet.isShowing -> origin.sheet.showResult(result.ok, message)
            origin is Origin.FromList && origin.sheet.sheet.isShowing -> {
                origin.sheet.showMessage(message)
                // The item action changed the list (e.g. Forget): fetch it again.
                if (result.ok) with(origin.sheet) { send(pipelineId, sourceAction, sourceArgs, Origin.Main) }
            }
            else -> messages.showMessage(message, !result.ok)
        }
    }

    private fun showList(pipeline: String, result: ServerMessage.CommandResult) {
        val open = listSheet?.takeIf { it.sheet.isShowing && it.pipelineId == pipeline && it.sourceAction == result.action }
        if (open != null) {
            open.bind(result)
            return
        }
        listSheet?.sheet?.dismiss()
        val spec = catalog.find(pipeline)
        val title = spec?.actions?.firstOrNull { it.id == result.action }?.label ?: result.action
        lateinit var sheet: ItemListSheet
        sheet = ItemListSheet(context, pipeline, result.action, emptyMap(), title) { item, action ->
            val run = { send(pipeline, action.id, mapOf("key" to OptionValue.Str(item.key)), Origin.FromList(sheet)) }
            if (!action.confirm) {
                run()
            } else {
                MaterialAlertDialogBuilder(context)
                    .setTitle(context.getString(R.string.item_confirm_title, action.label, item.label))
                    .setMessage(R.string.item_confirm_message)
                    .setPositiveButton(action.label) { _, _ -> run() }
                    .setNegativeButton(R.string.command_cancel, null)
                    .show()
            }
        }
        sheet.bind(result)
        sheet.sheet.onDismiss { if (listSheet === sheet) listSheet = null }
        listSheet = sheet
        sheet.sheet.show()
    }

    /** False when not connected (the caller says so where the user is looking). */
    private fun send(pipeline: String, action: String, args: Map<String, OptionValue>, origin: Origin): Boolean {
        val req = sendCommand(pipeline, action, args)
        if (req == null) {
            if (origin == Origin.Main) messages.showMessage(context.getString(R.string.command_not_connected), true)
            return false
        }
        pending[req] = Pending(pipeline, origin)
        return true
    }

    private companion object {
        const val CAPTURE_DIM = 0.15f
    }
}
