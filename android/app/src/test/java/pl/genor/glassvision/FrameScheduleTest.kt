package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Test

class FrameScheduleTest {
    /** Simulates a 30 fps camera (33 ms frames) with a 10 fps cap over 3 s. */
    @Test
    fun thirtyFpsCameraWithTenFpsCapSendsTenPerSecond() {
        var due = 0L
        var sent = 0
        for (i in 0 until 90) {
            val now = 1000L + i * 1000L / 30
            if (now >= due) {
                due = FrameAnalyzer.nextDue(due, now, 100)
                sent++
            }
        }
        assertEquals(30, sent)
    }

    @Test
    fun stallRestartsScheduleInsteadOfBursting() {
        assertEquals(5_100L, FrameAnalyzer.nextDue(previousDue = 1_000L, now = 5_000L, intervalMs = 100))
        assertEquals(1_100L, FrameAnalyzer.nextDue(previousDue = 1_000L, now = 1_033L, intervalMs = 100))
    }
}
