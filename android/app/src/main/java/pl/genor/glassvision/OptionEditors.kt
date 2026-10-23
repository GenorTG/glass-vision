package pl.genor.glassvision

import android.content.Context
import android.text.InputType
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.LinearLayout
import android.widget.TextView
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.slider.Slider
import com.google.android.material.textfield.MaterialAutoCompleteTextView
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout

/** A widget editing one [OptionSpec]-typed value. */
interface OptionEditor {
    val view: View

    /** The value currently shown (text not yet applied included). */
    fun value(): OptionValue

    /** Applies pending text input (IME done / focus loss / the ✓ icon do it too). */
    fun commit()
}

/**
 * Material rows for typed values, used by pipeline options, action arguments and the app's own settings:
 * bool → switch, bounded number → slider with value label, choice → exposed dropdown, text (and unbounded
 * number) → outlined field applied on IME done / focus loss / ✓, never per keystroke. With `onChange` null
 * nothing is applied live and callers read [OptionEditor.value] (action argument sheets).
 */
class OptionEditors(private val context: Context) {
    private val inflater = LayoutInflater.from(context)

    fun create(spec: OptionSpec, initial: OptionValue, onChange: ((OptionValue) -> Unit)?): OptionEditor =
        when {
            spec.type == OptionType.BOOL ->
                switch(spec.label, null, (initial as? OptionValue.Bool)?.value ?: false, onChange?.let { f -> { f(OptionValue.Bool(it)) } })
            spec.type == OptionType.CHOICE ->
                dropdown(spec.label, spec.choices, { it }, (initial as? OptionValue.Str)?.value, onChange)
            spec.isSlider -> {
                val steps = NumberSteps(spec.min!!, spec.max!!, spec.step)
                slider(spec.label, steps, (initial as? OptionValue.Num)?.value ?: spec.min, steps::format, onChange?.let { f -> { f(OptionValue.Num(it)) } })
            }
            else -> text(spec, initial, onChange)
        }

    fun switch(label: String, detail: String?, initial: Boolean, onChange: ((Boolean) -> Unit)?): OptionEditor {
        val row = inflate(R.layout.row_switch)
        row.findViewById<TextView>(R.id.row_label).text = label
        row.findViewById<TextView>(R.id.row_detail).apply {
            text = detail
            visibility = if (detail.isNullOrBlank()) View.GONE else View.VISIBLE
        }
        val switch = row.findViewById<MaterialSwitch>(R.id.row_switch)
        switch.isChecked = initial
        switch.contentDescription = label
        switch.setOnCheckedChangeListener { _, checked -> onChange?.invoke(checked) }
        row.setOnClickListener { switch.toggle() }
        return object : OptionEditor {
            override val view: View = row
            override fun value() = OptionValue.Bool(switch.isChecked)
            override fun commit() = Unit
        }
    }

    /** Slider over [steps]' integer positions; [format] renders the value label (units included). */
    fun slider(label: String, steps: NumberSteps, initial: Double, format: (Double) -> String, onChange: ((Double) -> Unit)?): OptionEditor {
        val row = inflate(R.layout.row_slider)
        row.findViewById<TextView>(R.id.row_label).text = label
        val valueText = row.findViewById<TextView>(R.id.row_value)
        val slider = row.findViewById<Slider>(R.id.row_slider)
        var current = initial
        valueText.text = format(current)
        slider.valueTo = steps.count.toFloat()
        slider.value = steps.toProgress(current).toFloat()
        slider.contentDescription = label
        slider.addOnChangeListener { _, v, fromUser ->
            if (!fromUser) return@addOnChangeListener
            current = steps.fromProgress(v.toInt())
            valueText.text = format(current)
            onChange?.invoke(current)
        }
        return object : OptionEditor {
            override val view: View = row
            override fun value() = OptionValue.Num(current)
            override fun commit() = Unit
        }
    }

    /** Exposed dropdown; [display] turns a choice into its menu text. */
    fun dropdown(
        label: String,
        choices: List<String>,
        display: (String) -> String,
        selected: String?,
        onChange: ((OptionValue) -> Unit)?,
    ): OptionEditor {
        val layout = inflate(R.layout.row_dropdown) as TextInputLayout
        layout.hint = label
        val field = layout.findViewById<MaterialAutoCompleteTextView>(R.id.row_dropdown_text)
        val shown = choices.map(display)
        field.setSimpleItems(shown.toTypedArray())
        var current = selected?.takeIf { it in choices } ?: choices.firstOrNull() ?: ""
        field.setText(if (current.isEmpty()) "" else display(current), false)
        field.setOnItemClickListener { _, _, position, _ ->
            current = choices[position]
            onChange?.invoke(OptionValue.Str(current))
        }
        return object : OptionEditor {
            override val view: View = layout
            override fun value() = OptionValue.Str(current)
            override fun commit() = Unit
        }
    }

    /** Full-width segmented buttons for 2–4 short choices. */
    fun segmented(label: String, choices: List<String>, display: (String) -> String, selected: String, onChange: (String) -> Unit): OptionEditor {
        val row = inflate(R.layout.row_segmented)
        row.findViewById<TextView>(R.id.row_label).text = label
        val group = row.findViewById<MaterialButtonToggleGroup>(R.id.row_toggle)
        val ids = choices.map { choice ->
            val button = inflater.inflate(R.layout.item_segment_button, group, false) as MaterialButton
            button.id = View.generateViewId()
            button.text = display(choice)
            group.addView(button)
            button.id
        }
        var current = selected.takeIf { it in choices } ?: choices.first()
        group.check(ids[choices.indexOf(current)])
        group.addOnButtonCheckedListener { _, id, checked ->
            if (!checked) return@addOnButtonCheckedListener
            val choice = choices[ids.indexOf(id)]
            if (choice != current) {
                current = choice
                onChange(choice)
            }
        }
        return object : OptionEditor {
            override val view: View = row
            override fun value() = OptionValue.Str(current)
            override fun commit() = Unit
        }
    }

    /** Text, or a number without bounds; with [onChange] a ✓ end icon applies it. */
    private fun text(spec: OptionSpec, initial: OptionValue, onChange: ((OptionValue) -> Unit)?): OptionEditor {
        val isNumber = spec.type == OptionType.NUMBER
        val parse = { s: String ->
            if (isNumber) s.trim().toDoubleOrNull()?.let { spec.coerce(OptionValue.Num(it)) } else OptionValue.Str(s)
        }
        val initialText = when (initial) {
            is OptionValue.Str -> initial.value
            is OptionValue.Num -> NumberSteps.plain(initial.value)
            is OptionValue.Bool -> initial.value.toString()
        }
        val inputType = if (isNumber) {
            InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL or InputType.TYPE_NUMBER_FLAG_SIGNED
        } else if (spec.key == "name") {
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PERSON_NAME or InputType.TYPE_TEXT_FLAG_CAP_WORDS
        } else {
            InputType.TYPE_CLASS_TEXT
        }
        var committed: OptionValue = initial
        lateinit var field: TextInputEditText
        val apply = {
            val v = parse(field.text?.toString().orEmpty())
            if (v != null && v != committed) {
                committed = v
                onChange?.invoke(v)
            }
        }
        val layout = textField(spec.label, initialText, inputType, null, if (onChange != null) apply else null)
        field = layout.findViewById(R.id.row_text_input)
        return object : OptionEditor {
            override val view: View = layout
            override fun value(): OptionValue = parse(field.text?.toString().orEmpty()) ?: committed
            override fun commit() = apply()
        }
    }

    /**
     * Outlined text field; with [onApply] it gets a ✓ end icon and applies on IME done and focus loss too.
     * Returned layout's input is `R.id.row_text_input`.
     */
    fun textField(label: String, initial: String, inputType: Int, helper: String?, onApply: (() -> Unit)?): TextInputLayout {
        val layout = inflate(R.layout.row_text) as TextInputLayout
        layout.hint = label
        helper?.let { layout.helperText = it }
        val field = layout.findViewById<TextInputEditText>(R.id.row_text_input)
        field.inputType = inputType
        field.setText(initial)
        if (onApply != null) {
            layout.endIconMode = TextInputLayout.END_ICON_CUSTOM
            layout.setEndIconDrawable(R.drawable.ic_check)
            layout.endIconContentDescription = context.getString(R.string.recognition_apply)
            layout.setEndIconOnClickListener {
                onApply()
                field.clearFocus()
            }
            field.setOnEditorActionListener { _, actionId, _ ->
                if (actionId == EditorInfo.IME_ACTION_DONE) onApply()
                false
            }
            field.setOnFocusChangeListener { _, hasFocus -> if (!hasFocus) onApply() }
        }
        return layout
    }

    /** Rows go into vertical LinearLayouts; XML margins need a parent at inflate time, so set them here. */
    private fun inflate(layout: Int): View = inflater.inflate(layout, null as ViewGroup?, false).apply {
        val field = this is TextInputLayout
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            .apply {
                topMargin = if (field) dp(12) else 0
                bottomMargin = if (field) dp(4) else 0
            }
    }

    private fun dp(v: Int): Int = (v * context.resources.displayMetrics.density).toInt()
}
