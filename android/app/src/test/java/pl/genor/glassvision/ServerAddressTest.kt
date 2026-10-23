package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ServerAddressTest {
    private fun n(input: String) = ServerAddress.normalize(input)

    @Test
    fun bare_host_and_host_port() {
        assertEquals("ws://192.168.1.50:8770/ws", n("192.168.1.50"))
        assertEquals("ws://192.168.1.50:9000/ws", n("192.168.1.50:9000"))
        assertEquals("ws://my-pc.local:8770/ws", n("My-PC.local"))
        assertEquals("ws://localhost:8770/ws", n("  localhost  "))
        assertEquals("ws://vision.example.com:8770/ws", n("vision.example.com/"))
    }

    @Test
    fun http_and_ws_schemes() {
        assertEquals("ws://192.168.1.50:8770/ws", n("http://192.168.1.50"))
        assertEquals("ws://192.168.1.50:8770/ws", n("http://192.168.1.50:8770/"))
        assertEquals("ws://192.168.1.50:8770/ws", n("ws://192.168.1.50:8770/ws"))
        assertEquals("ws://192.168.1.50:8770/ws", n("WS://192.168.1.50:8770/ws/"))
        assertEquals("ws://192.168.1.50:8000/ws", n("http://192.168.1.50:8000"))
    }

    /** TLS without a port = a reverse proxy on 443, so no :8770 is invented. */
    @Test
    fun tls_schemes() {
        assertEquals("wss://vision.example.com/ws", n("https://vision.example.com"))
        assertEquals("wss://vision.example.com/ws", n("wss://vision.example.com/ws"))
        assertEquals("wss://vision.example.com:8443/ws", n("https://vision.example.com:8443/"))
    }

    @Test
    fun reverse_proxy_prefix_is_kept() {
        assertEquals("wss://example.com/glass/ws", n("https://example.com/glass"))
        assertEquals("wss://example.com/glass/ws", n("https://example.com/glass/"))
        assertEquals("wss://example.com/glass/ws", n("wss://example.com/glass/ws"))
    }

    @Test
    fun ipv6_literals() {
        assertEquals("ws://[fd7a:115c::1]:8770/ws", n("[fd7a:115c::1]"))
        assertEquals("ws://[fd7a:115c::1]:9000/ws", n("http://[fd7a:115c::1]:9000"))
        assertEquals("ws://[::1]:8770/ws", n("::1"))
    }

    @Test
    fun query_and_fragment() {
        assertEquals(ServerAddress.Parsed("ws://192.168.1.50:8770/ws", "s3cret"), ServerAddress.parse("192.168.1.50:8770/ws?token=s3cret"))
        assertEquals(ServerAddress.Parsed("ws://h:8770/ws", "a b&c"), ServerAddress.parse("ws://h:8770/ws?x=1&token=a%20b%26c#frag"))
        assertEquals(ServerAddress.Parsed("ws://h:8770/ws", null), ServerAddress.parse("h?foo=bar"))
    }

    @Test
    fun rejects_unusable_input() {
        for (bad in listOf(
            "", "   ", "ftp://host", "host:0", "host:65536", "host:abc", "host:", "user@host", "http://",
            "two words", "-bad.example", "bad_host", "[fd7a::1", "[nothex]:80", "host:80:90x",
        )) {
            assertNull("'$bad' should be rejected", n(bad))
        }
    }

    @Test
    fun token_goes_into_the_socket_url_encoded() {
        assertEquals("ws://h:8770/ws", ServerEndpoint("ws://h:8770/ws", "").socketUrl())
        assertEquals("ws://h:8770/ws", ServerEndpoint("ws://h:8770/ws", "   ").socketUrl())
        assertEquals("ws://h:8770/ws?token=abc123", ServerEndpoint("ws://h:8770/ws", "abc123").socketUrl())
        assertEquals("ws://h:8770/ws?token=a%2Bb%2Fc%3D%26d", ServerEndpoint("ws://h:8770/ws", "a+b/c=&d").socketUrl())
        assertEquals("ws://h:8770/ws?token=two+words", ServerEndpoint("ws://h:8770/ws", "two words").socketUrl())
        assertEquals("ws://h:8770/ws?token=z%C3%B3%C5%82w", ServerEndpoint("ws://h:8770/ws", "zółw").socketUrl())
        assertEquals("ws://h/ws?x=1&token=t", ServerEndpoint.withToken("ws://h/ws?x=1", "t"))
    }

    @Test
    fun http_side_of_an_endpoint() {
        val e = ServerEndpoint("wss://example.com/glass/ws", " tok ")
        assertEquals("https://example.com/glass", e.httpBase)
        assertEquals("Bearer tok", e.authorization())
        assertNull(ServerEndpoint("ws://h:8770/ws").authorization())
        assertTrue(e.isConfigured)
        assertFalse(ServerEndpoint("").isConfigured)
        assertFalse(StreamSettings().endpoint.isConfigured) // first run: nothing to connect to
    }

    @Test
    fun auth_rejection_stops_reconnecting() {
        assertEquals(ConnectionLoss.Reaction.AUTH_FAILED, ConnectionLoss.reaction(4401, null))
        assertEquals(ConnectionLoss.Reaction.AUTH_FAILED, ConnectionLoss.reaction(null, 401))
        assertEquals(ConnectionLoss.Reaction.AUTH_FAILED, ConnectionLoss.reaction(null, 403))
        assertEquals(ConnectionLoss.Reaction.RECONNECT, ConnectionLoss.reaction(1000, null))
        assertEquals(ConnectionLoss.Reaction.RECONNECT, ConnectionLoss.reaction(1011, null))
        assertEquals(ConnectionLoss.Reaction.RECONNECT, ConnectionLoss.reaction(null, 502))
        assertEquals(ConnectionLoss.Reaction.RECONNECT, ConnectionLoss.reaction(null, null))
    }

    @Test
    fun settings_keep_the_server_across_reset() {
        val stream = SnapshotFixtures.MemoryStream(
            StreamSettings(serverUrl = "ws://192.168.1.50:8770/ws", token = " t ", targetFps = 3),
        )
        stream.resetToDefaults()
        assertEquals(ServerEndpoint("ws://192.168.1.50:8770/ws", "t"), stream.load().endpoint)
        assertEquals(StreamSettings().targetFps, stream.load().targetFps)
    }
}
