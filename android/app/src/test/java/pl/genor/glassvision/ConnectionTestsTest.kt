package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ConnectionTestsTest {
    private val info = HealthInfo("cuda:0 (NVIDIA GeForce RTX 3090)", "0.6.0.1")

    @Test
    fun parses_health() {
        assertEquals(info, ConnectionTests.parseHealth("""{"ok":true,"device":"cuda:0 (NVIDIA GeForce RTX 3090)","version":"0.6.0.1"}"""))
        assertNull(ConnectionTests.parseHealth("""{"ok":false}"""))
        assertNull(ConnectionTests.parseHealth("<html>nginx</html>"))
        assertNull(ConnectionTests.parseHealth(""))
    }

    @Test
    fun classifies_test_outcomes() {
        assertEquals(ConnectionTestResult.Ok(info), ConnectionTests.classify(200, info, 200, tokenGiven = true))
        // No APK served is not a token problem.
        assertEquals(ConnectionTestResult.Ok(info), ConnectionTests.classify(200, info, 404, tokenGiven = false))
        assertEquals(ConnectionTestResult.Ok(info), ConnectionTests.classify(200, info, null, tokenGiven = false))
        assertEquals(ConnectionTestResult.WrongToken(info, true), ConnectionTests.classify(200, info, 401, tokenGiven = true))
        assertEquals(ConnectionTestResult.WrongToken(info, false), ConnectionTests.classify(200, info, 403, tokenGiven = false))
        assertEquals(ConnectionTestResult.NotGlassVision("HTTP 404"), ConnectionTests.classify(404, null, null, tokenGiven = false))
    }

    @Test
    fun status_pill_routes_setup_states_to_the_connection_sheet() {
        assertTrue(StatusText.needsSetup(ConnectionState.NOT_CONFIGURED))
        assertTrue(StatusText.needsSetup(ConnectionState.UNAUTHORIZED))
        assertFalse(StatusText.needsSetup(ConnectionState.DISCONNECTED))
        assertFalse(StatusText.needsSetup(ConnectionState.CONNECTED))
        val idle = StatsSnapshot(0f, 0f, null, null, emptyMap(), 0, 0, 0)
        assertEquals(StatusPillModel("Not configured", Tone.IDLE), StatusText.pill(ConnectionState.NOT_CONFIGURED, false, idle))
        assertEquals(StatusPillModel("Wrong access token", Tone.BAD), StatusText.pill(ConnectionState.UNAUTHORIZED, false, idle))
        assertEquals(ChipBanner.NOT_CONFIGURED, StatusText.banner(ConnectionState.NOT_CONFIGURED, null))
        assertEquals(ChipBanner.UNAUTHORIZED, StatusText.banner(ConnectionState.UNAUTHORIZED, null))
    }
}
