package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class UpdateCheckTest {
    @Test
    fun http_base_from_ws_url() {
        assertEquals("http://192.168.1.50:8770", UpdateCheck.httpBase("ws://192.168.1.50:8770/ws"))
        assertEquals("https://box.example:443", UpdateCheck.httpBase(" WSS://box.example:443/ws?x=1 "))
        assertEquals("http://host", UpdateCheck.httpBase("ws://host"))
        // Reverse-proxy prefix stays, the socket path goes.
        assertEquals("https://box.example/glass", UpdateCheck.httpBase("wss://box.example/glass/ws"))
        assertEquals("http://host:8770", UpdateCheck.httpBase("ws://host:8770/ws?token=abc"))
        assertNull(UpdateCheck.httpBase("http://host/ws"))
        assertNull(UpdateCheck.httpBase("ws:///ws"))
    }

    @Test
    fun parses_server_version_json() {
        val info = UpdateCheck.parse(
            """{"application_id":"pl.genor.glassvision","version_code":3000001,"version_name":"0.3.0.1",
               "size":4642761,"sha256":"ABCDEF","url":"/app.apk"}"""
        )
        assertEquals(UpdateInfo("pl.genor.glassvision", 3000001, "0.3.0.1", 4642761, "abcdef", "/app.apk"), info)
        assertNull(UpdateCheck.parse("""{"version_code":1}"""))
        assertNull(UpdateCheck.parse("<html>"))
    }

    @Test
    fun only_newer_builds_of_this_app_are_updates() {
        val info = UpdateInfo("pl.genor.glassvision", 3000002, "0.3.0.2", 1, "x", "/app.apk")
        assertTrue(UpdateCheck.isUpdate(info, 3000001, "pl.genor.glassvision"))
        assertFalse(UpdateCheck.isUpdate(info, 3000002, "pl.genor.glassvision"))
        assertFalse(UpdateCheck.isUpdate(info, 3000003, "pl.genor.glassvision"))
        assertFalse(UpdateCheck.isUpdate(info, 1, "com.other.app"))
    }
}
