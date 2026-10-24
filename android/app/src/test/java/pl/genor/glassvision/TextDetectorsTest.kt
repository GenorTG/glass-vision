package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Test

/** Data detectors on read text: what becomes a call / link / e-mail / calendar action. */
class TextDetectorsTest {
    private fun kinds(text: String) = TextDetectors.entities(text).map { it.kind to it.value }

    @Test
    fun phones() {
        assertEquals(listOf(DetectedKind.PHONE to "+48123456789"), kinds("Tel. +48 123 456 789"))
        assertEquals(listOf(DetectedKind.PHONE to "+493012345678"), kinds("Hotline 0049 (30) 1234-5678"))
        assertEquals(listOf(DetectedKind.PHONE to "0301234567"), kinds("Info: 030 123 4567"))
        // Not phones: times, prices, years, short codes, a year range.
        assertEquals(emptyList<Pair<DetectedKind, String>>(), kinds("Abfahrt 12:45 · 4,50 € · seit 1998 · Gleis 12 · 1998-2024"))
    }

    @Test
    fun links_and_mail() {
        assertEquals(listOf(DetectedKind.URL to "https://www.bahn.de/info"), kinds("Mehr: www.bahn.de/info."))
        assertEquals(listOf(DetectedKind.URL to "http://example.org/a?b=1"), kinds("(http://example.org/a?b=1)"))
        assertEquals(listOf(DetectedKind.URL to "https://zoo-wroclaw.pl"), kinds("zoo-wroclaw.pl"))
        assertEquals(listOf(DetectedKind.EMAIL to "kontakt@zoo-wroclaw.pl"), kinds("Napisz: kontakt@zoo-wroclaw.pl"))
        assertEquals(emptyList<Pair<DetectedKind, String>>(), kinds("Notausgang 2. Bitte freihalten"))
    }

    @Test
    fun dates() {
        assertEquals(listOf(DetectedKind.DATE to "2026-10-24"), kinds("Gültig bis 24.10.2026"))
        assertEquals(listOf(DetectedKind.DATE to "2026-03-05"), kinds("05/03/26"))
        assertEquals(listOf(DetectedKind.DATE to "2026-12-25"), kinds("12/25/2026")) // US order when the month can't be 25
        assertEquals(listOf(DetectedKind.DATE to "2026-01-31"), kinds("2026-01-31"))
        assertEquals(emptyList<Pair<DetectedKind, String>>(), kinds("31.13.2026 · 1.5.3"))
    }

    @Test
    fun several_in_order_with_their_text() {
        val found = TextDetectors.entities("Anmeldung 01.11.2026 unter +48 600 700 800 oder info@kino.pl, www.kino.pl")
        assertEquals(listOf(DetectedKind.DATE, DetectedKind.PHONE, DetectedKind.EMAIL, DetectedKind.URL), found.map { it.kind })
        assertEquals(listOf("01.11.2026", "+48 600 700 800", "info@kino.pl", "www.kino.pl"), found.map { it.text })
    }
}
