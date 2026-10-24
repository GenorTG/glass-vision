package pl.genor.glassvision

import android.content.Context
import android.view.OrientationEventListener
import android.view.Surface

/**
 * Device orientation → the Read still's `ImageCapture.targetRotation`. The activity is locked to portrait, so the
 * display rotation never changes and CameraX would make every still portrait-upright: a sign shot with the phone
 * held sideways came out 90° off. CameraX's recipe for a locked orientation: an [OrientationEventListener] sets the
 * target rotation of the use cases except Preview
 * (https://developer.android.com/media/camera/camerax/orientation-rotation). Only the still follows it here; the live
 * analysis frames stay portrait (the overlay and BoxMapper assume that).
 */
object CaptureRotation {
    /** Degrees past a sector boundary before the rotation flips, so a phone held near 45° doesn't flicker. */
    const val HYSTERESIS_DEG = 10

    /**
     * The target rotation for [degrees] (OrientationEventListener: clockwise from natural, or
     * [OrientationEventListener.ORIENTATION_UNKNOWN] when the phone lies flat), starting from [current]: unknown keeps
     * [current] (a page shot from above keeps the last way the phone was held), and so does any angle within
     * 45° + [hysteresis] of [current]'s centre.
     */
    fun next(current: Int, degrees: Int, hysteresis: Int = HYSTERESIS_DEG): Int {
        if (degrees < 0) return current
        val d = degrees % 360
        val diff = Math.floorMod(d - centreOf(current), 360)
        if (minOf(diff, 360 - diff) <= 45 + hysteresis) return current
        return of(d)
    }

    /** CameraX docs' mapping: the phone turned clockwise by 90° (landscape, top to the right) = ROTATION_270. */
    fun of(degrees: Int): Int = when (Math.floorMod(degrees, 360)) {
        in 45 until 135 -> Surface.ROTATION_270
        in 135 until 225 -> Surface.ROTATION_180
        in 225 until 315 -> Surface.ROTATION_90
        else -> Surface.ROTATION_0
    }

    /** Orientation degrees at the middle of [rotation]'s sector. */
    fun centreOf(rotation: Int): Int = when (rotation) {
        Surface.ROTATION_270 -> 90
        Surface.ROTATION_180 -> 180
        Surface.ROTATION_90 -> 270
        else -> 0
    }

    /** [rotation] (Surface.ROTATION_*) in degrees, as CameraX counts target rotations. */
    fun degreesOf(rotation: Int): Int = when (rotation) {
        Surface.ROTATION_90 -> 90
        Surface.ROTATION_180 -> 180
        Surface.ROTATION_270 -> 270
        else -> 0
    }
}

/**
 * Feeds [CaptureRotation.next] from the orientation sensor; [onRotation] (main thread) only on a change. [enable] in
 * onStart, [disable] in onStop (the CameraX recipe); the last rotation is kept across both.
 */
class CaptureOrientation(context: Context, private val onRotation: (Int) -> Unit) {
    var rotation: Int = Surface.ROTATION_0
        private set

    private val listener = object : OrientationEventListener(context) {
        override fun onOrientationChanged(orientation: Int) {
            val next = CaptureRotation.next(rotation, orientation)
            if (next != rotation) {
                rotation = next
                onRotation(next)
            }
        }
    }

    fun enable() {
        if (listener.canDetectOrientation()) listener.enable()
    }

    fun disable() = listener.disable()
}
