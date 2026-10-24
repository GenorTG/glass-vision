package pl.genor.glassvision

import android.view.OrientationEventListener
import android.view.Surface
import org.junit.Assert.assertEquals
import org.junit.Test

class CaptureRotationTest {
    @Test
    fun maps_orientation_degrees_like_the_camerax_docs() {
        assertEquals(Surface.ROTATION_0, CaptureRotation.of(0))
        assertEquals(Surface.ROTATION_0, CaptureRotation.of(44))
        assertEquals(Surface.ROTATION_270, CaptureRotation.of(45)) // turned clockwise: left side up
        assertEquals(Surface.ROTATION_270, CaptureRotation.of(134))
        assertEquals(Surface.ROTATION_180, CaptureRotation.of(180))
        assertEquals(Surface.ROTATION_90, CaptureRotation.of(270)) // turned counter-clockwise: right side up
        assertEquals(Surface.ROTATION_0, CaptureRotation.of(315))
        assertEquals(Surface.ROTATION_0, CaptureRotation.of(359))
    }

    @Test
    fun a_sideways_phone_rotates_the_still() {
        assertEquals(Surface.ROTATION_90, CaptureRotation.next(Surface.ROTATION_0, 270))
        assertEquals(Surface.ROTATION_270, CaptureRotation.next(Surface.ROTATION_0, 90))
        assertEquals(Surface.ROTATION_180, CaptureRotation.next(Surface.ROTATION_0, 180))
        assertEquals(Surface.ROTATION_0, CaptureRotation.next(Surface.ROTATION_90, 0))
    }

    @Test
    fun flat_phone_keeps_the_last_rotation() {
        val unknown = OrientationEventListener.ORIENTATION_UNKNOWN
        assertEquals(Surface.ROTATION_90, CaptureRotation.next(Surface.ROTATION_90, unknown))
        assertEquals(Surface.ROTATION_0, CaptureRotation.next(Surface.ROTATION_0, unknown))
    }

    @Test
    fun hysteresis_holds_near_the_45_degree_boundaries() {
        // Portrait until 10° past 45 / 315, then the new sector.
        assertEquals(Surface.ROTATION_0, CaptureRotation.next(Surface.ROTATION_0, 50))
        assertEquals(Surface.ROTATION_0, CaptureRotation.next(Surface.ROTATION_0, 55))
        assertEquals(Surface.ROTATION_270, CaptureRotation.next(Surface.ROTATION_0, 56))
        assertEquals(Surface.ROTATION_0, CaptureRotation.next(Surface.ROTATION_0, 305))
        assertEquals(Surface.ROTATION_90, CaptureRotation.next(Surface.ROTATION_0, 304))
        // And back: landscape (centre 270) holds until 270 + 55.
        assertEquals(Surface.ROTATION_90, CaptureRotation.next(Surface.ROTATION_90, 320))
        assertEquals(Surface.ROTATION_90, CaptureRotation.next(Surface.ROTATION_90, 216))
        assertEquals(Surface.ROTATION_0, CaptureRotation.next(Surface.ROTATION_90, 330))
        assertEquals(Surface.ROTATION_180, CaptureRotation.next(Surface.ROTATION_90, 210))
    }

    @Test
    fun a_slow_turn_flips_once() {
        var r = Surface.ROTATION_0
        val seen = mutableListOf(r)
        for (d in (0..100) + (100 downTo 40)) {
            val next = CaptureRotation.next(r, d)
            if (next != r) seen += next
            r = next
        }
        // Wobbling between 40 and 100 after the flip doesn't flip back above 35° (90 − 55).
        assertEquals(listOf(Surface.ROTATION_0, Surface.ROTATION_270), seen)
    }

    @Test
    fun degrees_of_rotation() {
        assertEquals(0, CaptureRotation.degreesOf(Surface.ROTATION_0))
        assertEquals(90, CaptureRotation.degreesOf(Surface.ROTATION_90))
        assertEquals(180, CaptureRotation.degreesOf(Surface.ROTATION_180))
        assertEquals(270, CaptureRotation.degreesOf(Surface.ROTATION_270))
    }

    @Test
    fun still_turns_against_the_phone_on_the_back_camera_and_with_it_on_the_front() {
        fun shot(facing: CameraFacing, target: Int) = CameraBinder.Shot(ByteArray(0), 0, target, facing)
        assertEquals(0, shot(CameraFacing.BACK, 0).analysisToStillDegrees)
        // Phone turned clockwise (ROTATION_270): the portrait frame shows the text turned counter-clockwise,
        // so the still is the live frame turned clockwise by 90.
        assertEquals(90, shot(CameraFacing.BACK, 270).analysisToStillDegrees)
        assertEquals(270, shot(CameraFacing.BACK, 90).analysisToStillDegrees)
        assertEquals(180, shot(CameraFacing.BACK, 180).analysisToStillDegrees)
        assertEquals(270, shot(CameraFacing.FRONT, 270).analysisToStillDegrees)
        assertEquals(90, shot(CameraFacing.FRONT, 90).analysisToStillDegrees)
    }
}
