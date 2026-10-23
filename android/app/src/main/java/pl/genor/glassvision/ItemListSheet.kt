package pl.genor.glassvision

import android.content.Context
import android.content.res.ColorStateList
import android.view.LayoutInflater
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.snackbar.Snackbar

/**
 * A command_result with `items` (e.g. Faces → Known faces) as a bottom-sheet list; each row offers the
 * result's `item_action` (e.g. Forget). [bind] replaces the content in place, so the list refreshes after an
 * item action without the sheet jumping. [pipelineId]/[sourceAction]/[sourceArgs] = the command that produced
 * the list, re-sent to refresh it. Main thread only.
 */
class ItemListSheet(
    private val context: Context,
    val pipelineId: String,
    val sourceAction: String,
    val sourceArgs: Map<String, OptionValue>,
    title: String,
    private val onItem: (ServerMessage.CommandResult.ResultItem, ServerMessage.CommandResult.ItemAction) -> Unit,
) {
    val sheet = Sheet(context)
    private val list = LinearLayout(context).apply { orientation = LinearLayout.VERTICAL }
    private val inflater = LayoutInflater.from(context)
    private val dp = context.resources.displayMetrics.density

    init {
        sheet.setTitle(title)
        sheet.setBody(list)
    }

    fun bind(result: ServerMessage.CommandResult) {
        sheet.setSubtitle(result.message.takeIf { result.items.isNotEmpty() })
        list.removeAllViews()
        if (result.items.isEmpty()) {
            list.addView(emptyState(result.message.ifEmpty { context.getString(R.string.command_ok, result.action) }))
            return
        }
        val action = result.itemAction
        for (item in result.items) {
            val row = inflater.inflate(R.layout.item_result_row, list, false)
            (row.layoutParams as LinearLayout.LayoutParams).bottomMargin = (8 * dp).toInt()
            row.findViewById<TextView>(R.id.row_avatar).text = item.label.trim().take(1).uppercase()
            row.findViewById<TextView>(R.id.row_label).text = item.label
            if (action != null) {
                row.findViewById<MaterialButton>(R.id.row_action).apply {
                    visibility = View.VISIBLE
                    text = action.label
                    contentDescription = context.getString(R.string.item_confirm_title, action.label, item.label)
                    if (action.confirm) {
                        // Destructive (the server asks to confirm): error colour + bin icon.
                        val red = ColorStateList.valueOf(ContextCompat.getColor(context, R.color.gv_error))
                        setTextColor(red)
                        iconTint = red
                        setIconResource(R.drawable.ic_delete)
                    }
                    setOnClickListener { onItem(item, action) }
                }
            }
            list.addView(row)
        }
    }

    fun showMessage(message: String) {
        if (sheet.isShowing) Snackbar.make(sheet.view, message, Snackbar.LENGTH_SHORT).show()
    }

    private fun emptyState(message: String) = TextView(context).apply {
        text = message
        setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodyLarge)
        setTextColor(ContextCompat.getColor(context, R.color.gv_on_surface_variant))
        gravity = android.view.Gravity.CENTER
        val pad = (24 * dp).toInt()
        setPadding(pad, pad, pad, pad * 2)
    }
}
