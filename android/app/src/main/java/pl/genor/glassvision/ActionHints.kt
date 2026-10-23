package pl.genor.glassvision

import androidx.annotation.DrawableRes
import androidx.annotation.StringRes

/**
 * How the app presents one action. The protocol only describes args, so this is the client's own presentation
 * layer for actions it knows well; every other action gets the generic look (label, args sheet, run button).
 */
data class ActionPresentation(
    @DrawableRes val icon: Int? = null,
    @StringRes val tip: Int? = null,
    /** Primary button text; null = the action's label from hello. */
    @StringRes val primaryLabel: Int? = null,
    /** Stay open after sending (repeatable captures); else close and show the result on the main screen. */
    val keepOpen: Boolean = false,
    /** Offer a one-tap switch to the front camera while on the back one (selfie enrolment). */
    val offerFrontCamera: Boolean = false,
)

object ActionHints {
    fun of(pipeline: String, action: String): ActionPresentation = when ("$pipeline/$action") {
        "faces/enroll" -> ActionPresentation(
            icon = R.drawable.ic_face,
            tip = R.string.learn_face_tip,
            primaryLabel = R.string.learn_face_capture,
            keepOpen = true,
            offerFrontCamera = true,
        )
        "faces/list" -> ActionPresentation(icon = R.drawable.ic_list)
        else -> ActionPresentation()
    }
}
