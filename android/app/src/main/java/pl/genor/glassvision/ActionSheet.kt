package pl.genor.glassvision

import android.content.Context
import android.content.res.ColorStateList
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.snackbar.Snackbar

/**
 * Bottom sheet for one pipeline action with arguments (e.g. Faces → Learn face): optional tip, optional
 * "switch to the front camera" offer, one editor per argument, a big primary button. With
 * [ActionPresentation.keepOpen] it stays open for repeated captures and shows each result inline + as a
 * snackbar; otherwise the caller closes it after sending. Plain data in, callbacks out. Main thread only.
 */
class ActionSheet(
    private val context: Context,
    val pipeline: PipelineSpec,
    val action: ActionSpec,
    initialArgs: Map<String, OptionValue>,
    facing: CameraFacing,
    private val frontAvailable: Boolean,
    private val onSwitchToFront: () -> Unit,
    private val onSubmit: (Map<String, OptionValue>) -> Unit,
) {
    val presentation = ActionHints.of(pipeline.id, action.id)
    val sheet = Sheet(context)
    private val body = LayoutInflater.from(context).inflate(R.layout.sheet_action_body, null as ViewGroup?, false)
    private val cameraOffer: View = body.findViewById(R.id.action_camera_offer)
    private val resultRow: View = body.findViewById(R.id.action_result)
    private val resultIcon: ImageView = body.findViewById(R.id.action_result_icon)
    private val resultText: TextView = body.findViewById(R.id.action_result_text)
    private val primary: MaterialButton = body.findViewById(R.id.action_primary)
    private val inputs: List<Pair<OptionSpec, OptionEditor>>

    init {
        sheet.setTitle(context.getString(R.string.action_sheet_title, pipeline.name, action.label))
        sheet.setBody(body)

        presentation.tip?.let {
            body.findViewById<View>(R.id.action_tip).visibility = View.VISIBLE
            body.findViewById<TextView>(R.id.action_tip_text).setText(it)
        }
        body.findViewById<View>(R.id.action_camera_switch).setOnClickListener { onSwitchToFront() }
        onCameraChanged(facing)

        val editors = OptionEditors(context)
        val box = body.findViewById<LinearLayout>(R.id.action_args)
        inputs = action.args.map { spec ->
            spec to editors.create(spec, initialArgs[spec.key] ?: spec.default, onChange = null).also { box.addView(it.view) }
        }

        primary.text = presentation.primaryLabel?.let(context::getString) ?: action.label
        primary.setIconResource(if (presentation.keepOpen) R.drawable.ic_capture else R.drawable.ic_check)
        primary.setOnClickListener { onSubmit(values()) }
    }

    /** The arguments as typed, coerced to their declared types. */
    fun values(): Map<String, OptionValue> = inputs.associate { (spec, editor) -> spec.key to spec.coerce(editor.value()) }

    /** The offer only makes sense on the back camera of a phone that has a front one. */
    fun onCameraChanged(facing: CameraFacing) {
        val offer = presentation.offerFrontCamera && frontAvailable && facing == CameraFacing.BACK
        cameraOffer.visibility = if (offer) View.VISIBLE else View.GONE
    }

    fun showSending() {
        primary.isEnabled = false
        showInline(context.getString(R.string.command_sent), null)
    }

    /** A command_result for this sheet: inline (stays, e.g. the sample count) and as a snackbar. */
    fun showResult(ok: Boolean, message: String, snackbar: Boolean = true) {
        primary.isEnabled = true
        showInline(message, ok)
        if (snackbar && sheet.isShowing) {
            Snackbar.make(sheet.view, message, Snackbar.LENGTH_SHORT).setAnchorView(primary).show()
        }
    }

    private fun showInline(message: String, ok: Boolean?) {
        resultRow.visibility = View.VISIBLE
        resultText.text = message
        val color = ContextCompat.getColor(
            context,
            when (ok) {
                true -> R.color.gv_status_good
                false -> R.color.gv_status_bad
                null -> R.color.gv_on_surface_variant
            },
        )
        resultIcon.setImageResource(if (ok == false) R.drawable.ic_error_dot else R.drawable.ic_check)
        resultIcon.imageTintList = if (ok == false) null else ColorStateList.valueOf(color)
        resultIcon.visibility = if (ok == null) View.INVISIBLE else View.VISIBLE
        resultText.setTextColor(if (ok == null) color else ContextCompat.getColor(context, R.color.gv_on_surface))
    }
}
