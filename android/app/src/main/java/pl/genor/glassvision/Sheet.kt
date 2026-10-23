package pl.genor.glassvision

import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.TextView
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog

/**
 * The one bottom-sheet look of the app (sheet_scaffold.xml): handle, optional title/subtitle, scrolling body. The view draws its own rounded surface, so it renders identically inside the dialog and in
 * snapshot tests; [show] wraps it in a [BottomSheetDialog]. Main thread only.
 */
class Sheet(val context: Context) {
    val view: View = LayoutInflater.from(context).inflate(R.layout.sheet_scaffold, null as ViewGroup?, false)
    private val header: View = view.findViewById(R.id.sheet_header)
    private val title: TextView = view.findViewById(R.id.sheet_title)
    private val subtitle: TextView = view.findViewById(R.id.sheet_subtitle)
    private val body: FrameLayout = view.findViewById(R.id.sheet_body)
    private var dialog: BottomSheetDialog? = null
    private val dismissListeners = mutableListOf<() -> Unit>()

    val isShowing: Boolean get() = dialog?.isShowing == true

    fun setTitle(text: CharSequence) {
        title.text = text
        header.visibility = View.VISIBLE
    }

    fun setSubtitle(text: CharSequence?) {
        subtitle.text = text
        subtitle.visibility = if (text.isNullOrBlank()) View.GONE else View.VISIBLE
    }

    fun setBody(content: View) {
        body.removeAllViews()
        body.addView(content, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
    }

    fun onDismiss(listener: () -> Unit) {
        dismissListeners += listener
    }

    /**
     * [expanded] = open at full height (long content); [dimAmount] < default keeps the camera visible behind
     * capture flows.
     */
    fun show(expanded: Boolean = false, dimAmount: Float? = null): Sheet {
        val d = BottomSheetDialog(context)
        d.setContentView(view)
        d.setOnDismissListener {
            dialog = null
            (view.parent as? ViewGroup)?.removeView(view)
            dismissListeners.toList().forEach { it() }
        }
        dimAmount?.let { d.window?.setDimAmount(it) }
        if (expanded) {
            d.behavior.skipCollapsed = true
            d.behavior.state = BottomSheetBehavior.STATE_EXPANDED
        }
        dialog = d
        d.show()
        return this
    }

    fun dismiss() {
        dialog?.dismiss()
    }
}
