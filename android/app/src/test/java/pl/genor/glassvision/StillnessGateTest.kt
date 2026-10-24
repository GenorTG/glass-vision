package pl.genor.glassvision

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class StillnessGateTest {
    @Test
    fun opens_after_the_phone_stays_calm_for_the_hold_time() {
        val gate = StillnessGate(startMs = 0)
        assertFalse(gate.onSample(0, 0.8f, 0f, 0f)) // the tap's jolt
        assertFalse(gate.onSample(40, 0.05f, 0.05f, 0f))
        assertFalse(gate.onSample(100, 0.05f, 0.02f, 0.01f))
        assertTrue(gate.onSample(140, 0.01f, 0f, 0.02f))
        assertTrue(gate.open)
    }

    @Test
    fun movement_restarts_the_calm_period() {
        val gate = StillnessGate(startMs = 0)
        assertFalse(gate.onSample(0, 0.05f, 0f, 0f))
        assertFalse(gate.onSample(90, 0f, 0.2f, 0.1f)) // |ω| = 0.22: moving again
        assertFalse(gate.onSample(100, 0.05f, 0f, 0f))
        assertFalse(gate.onSample(190, 0.05f, 0f, 0f))
        assertTrue(gate.onSample(200, 0.05f, 0f, 0f))
    }

    @Test
    fun a_shaky_hand_still_gets_a_picture_after_the_cap() {
        val gate = StillnessGate(startMs = 1000)
        var t = 1000L
        while (t < 1000 + StillnessGate.CAP_MS) {
            assertFalse(gate.onSample(t, 0.5f, 0.3f, 0f))
            t += 20
        }
        assertTrue(gate.onSample(t, 0.5f, 0.3f, 0f))
        assertFalse(gate.open) // released by the cap, not calm
        assertTrue(gate.expired(1000 + StillnessGate.CAP_MS))
    }

    @Test
    fun threshold_is_on_the_total_angular_speed() {
        val gate = StillnessGate(startMs = 0, holdMs = 0)
        // Each axis below 0.15, the magnitude (0.17) above.
        assertFalse(gate.onSample(0, 0.1f, 0.1f, 0.1f))
        assertTrue(gate.onSample(10, 0.1f, 0.1f, 0f))
    }
}
