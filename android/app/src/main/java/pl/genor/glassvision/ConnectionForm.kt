package pl.genor.glassvision

import android.content.Context
import android.content.res.ColorStateList
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout

/**
 * Server address + optional access token, "Test connection" and "Connect" (view_connection_form.xml). The address
 * is normalized live by [ServerAddress] and the helper line shows what will be connected to. Shared by the
 * first-run [ConnectionSheet] and Settings → Connection. [tester] null = no test button (snapshots pass results
 * in through [showResult]). Main thread only.
 */
class ConnectionForm(
    private val context: Context,
    initial: ServerEndpoint,
    private val tester: ((ServerEndpoint, (ConnectionTestResult) -> Unit) -> Unit)?,
    private val onConnect: (ServerEndpoint) -> Unit,
) {
    val view: View = LayoutInflater.from(context).inflate(R.layout.view_connection_form, null as ViewGroup?, false)
    private val addressLayout: TextInputLayout = view.findViewById(R.id.connection_address_layout)
    private val address: TextInputEditText = view.findViewById(R.id.connection_address)
    private val token: TextInputEditText = view.findViewById(R.id.connection_token)
    private val resultRow: View = view.findViewById(R.id.connection_result)
    private val resultIcon: ImageView = view.findViewById(R.id.connection_result_icon)
    private val resultText: TextView = view.findViewById(R.id.connection_result_text)
    private val testButton: MaterialButton = view.findViewById(R.id.connection_test)
    /** Bumped on every edit/test so a late result for older input is ignored. */
    private var testGeneration = 0

    init {
        address.setText(initial.wsUrl)
        token.setText(initial.token)
        updateHelper()
        address.addTextChangedListener(afterChange {
            updateHelper()
            clearResult()
        })
        token.addTextChangedListener(afterChange(::clearResult))
        testButton.visibility = if (tester != null) View.VISIBLE else View.GONE
        testButton.setOnClickListener { runTest() }
        view.findViewById<MaterialButton>(R.id.connection_connect).setOnClickListener { connect() }
    }

    fun showIntro() {
        view.findViewById<View>(R.id.connection_intro).visibility = View.VISIBLE
    }

    /** First run: offer recognition on this phone instead (no server needed). */
    fun showRunLocal(onRunLocal: () -> Unit) {
        view.findViewById<MaterialButton>(R.id.connection_run_local).apply {
            visibility = View.VISIBLE
            setOnClickListener { onRunLocal() }
        }
    }

    /** What the fields currently say, normalized; null while the address is unusable. */
    fun endpoint(): ServerEndpoint? {
        val parsed = ServerAddress.parse(address.text?.toString().orEmpty()) ?: return null
        // A token pasted inside the URL counts unless the token field says otherwise.
        val typed = token.text?.toString()?.trim().orEmpty()
        return ServerEndpoint(parsed.wsUrl, typed.ifEmpty { parsed.token.orEmpty() })
    }

    fun showTesting() = showInline(context.getString(R.string.connection_testing), null)

    fun showResult(result: ConnectionTestResult) {
        val ok = result is ConnectionTestResult.Ok
        val text = when (result) {
            is ConnectionTestResult.Ok ->
                context.getString(R.string.connection_ok, StatusText.serverInfo(result.info.version, result.info.device))
            is ConnectionTestResult.WrongToken -> context.getString(
                if (result.tokenGiven) R.string.connection_wrong_token else R.string.connection_needs_token,
                StatusText.serverInfo(result.info.version, result.info.device),
            )
            is ConnectionTestResult.NotGlassVision -> context.getString(R.string.connection_not_glass_vision, result.detail)
            is ConnectionTestResult.Unreachable -> context.getString(R.string.connection_unreachable, result.detail)
            ConnectionTestResult.InvalidAddress -> context.getString(R.string.connection_invalid)
        }
        showInline(text, ok)
    }

    /** A problem reported from outside the form (e.g. the server closed the socket with 4401). */
    fun showError(text: String) = showInline(text, false)

    private fun runTest() {
        val run = tester ?: return
        val endpoint = endpoint() ?: return showResult(ConnectionTestResult.InvalidAddress)
        val generation = ++testGeneration
        testButton.isEnabled = false
        showTesting()
        run(endpoint) { result ->
            if (generation != testGeneration) return@run
            testButton.isEnabled = true
            showResult(result)
        }
    }

    private fun connect() {
        val endpoint = endpoint() ?: return showResult(ConnectionTestResult.InvalidAddress)
        onConnect(endpoint)
    }

    private fun updateHelper() {
        val raw = address.text?.toString().orEmpty()
        val url = ServerAddress.normalize(raw)
        addressLayout.helperText = when {
            raw.isBlank() -> context.getString(R.string.connection_address_help)
            url == null -> context.getString(R.string.connection_invalid)
            else -> context.getString(R.string.connection_connects_to, url)
        }
    }

    private fun clearResult() {
        testGeneration++
        testButton.isEnabled = true
        resultRow.visibility = View.GONE
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

    private fun afterChange(block: () -> Unit) = object : TextWatcher {
        override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
        override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = Unit
        override fun afterTextChanged(s: Editable?) = block()
    }
}

/**
 * "Connect to your server": the first-run sheet (no server configured yet) and where a rejected token lands.
 * With [onRunLocal] it also offers "No server? Run on this phone". Main thread only.
 */
class ConnectionSheet(
    context: Context,
    initial: ServerEndpoint,
    tester: ((ServerEndpoint, (ConnectionTestResult) -> Unit) -> Unit)?,
    onRunLocal: (() -> Unit)? = null,
    onConnect: (ServerEndpoint) -> Unit,
) {
    val sheet = Sheet(context)
    val form = ConnectionForm(context, initial, tester) { endpoint ->
        onConnect(endpoint)
        sheet.dismiss()
    }

    init {
        sheet.setTitle(context.getString(R.string.connection_title))
        form.showIntro()
        onRunLocal?.let { run ->
            form.showRunLocal {
                run()
                sheet.dismiss()
            }
        }
        sheet.setBody(form.view)
    }
}
