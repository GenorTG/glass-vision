package pl.genor.glassvision

/*
 * Data detectors on read text (phone numbers, links, e-mail addresses, dates) → tappable actions in the Read text
 * sheet. Plain regexes instead of Linkify (its android.util.Patterns are stubs in JVM tests, no dates) or ML Kit
 * Entity Extraction (beta, a ~5.6 MB model download per language): signs carry few entities, in simple formats.
 */

enum class DetectedKind { PHONE, URL, EMAIL, DATE }

/**
 * An entity in a text: [text] as written, [value] normalized for its action (tel: digits with a leading +, a URL with
 * a scheme, the address, an ISO date yyyy-mm-dd).
 */
data class Detected(val kind: DetectedKind, val text: String, val value: String)

object TextDetectors {
    private val EMAIL = Regex("[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\\.[A-Za-z0-9-]+)*\\.[A-Za-z]{2,}")
    private val URL = Regex(
        "(?:https?://|www\\.)[^\\s]+|\\b[A-Za-z0-9-]+(?:\\.[A-Za-z0-9-]+)*\\.(?:com|org|net|info|io|eu|de|pl|at|ch|cz|sk|fr|it|es|pt|nl|be|uk|ie|dk|se|no|fi|gov|edu)(?:/[^\\s]*)?\\b",
        RegexOption.IGNORE_CASE,
    )
    private val PHONE = Regex("(?:\\+|\\b00)\\d[\\d ().\\-/]{5,}\\d|\\(?\\b\\d[\\d ().\\-/]{6,}\\d\\b")
    private val DATE_DMY = Regex("\\b(\\d{1,2})[./](\\d{1,2})[./](\\d{4}|\\d{2})\\b")
    private val DATE_ISO = Regex("\\b(\\d{4})-(\\d{2})-(\\d{2})\\b")
    private const val URL_TRAILING = ".,;:!?)]}'\""

    /** Entities in [text], in order of appearance, each once. */
    fun find(text: String): List<Pair<IntRange, Detected>> {
        val found = ArrayList<Pair<IntRange, Detected>>()
        fun free(r: IntRange) = found.none { (o, _) -> r.first <= o.last && o.first <= r.last }

        for (m in DATE_ISO.findAll(text)) date(m.groupValues[3], m.groupValues[2], m.groupValues[1])?.let { found += m.range to Detected(DetectedKind.DATE, m.value, it) }
        for (m in DATE_DMY.findAll(text)) {
            if (!free(m.range)) continue
            date(m.groupValues[1], m.groupValues[2], m.groupValues[3])?.let { found += m.range to Detected(DetectedKind.DATE, m.value, it) }
        }
        for (m in EMAIL.findAll(text)) if (free(m.range)) found += m.range to Detected(DetectedKind.EMAIL, m.value, m.value)
        for (m in URL.findAll(text)) {
            val value = m.value.trimEnd { it in URL_TRAILING }
            val range = m.range.first until m.range.first + value.length
            if (value.isEmpty() || !free(range) || '@' in value) continue
            val url = if (value.startsWith("http", ignoreCase = true)) value else "https://$value"
            found += range to Detected(DetectedKind.URL, value, url)
        }
        for (m in PHONE.findAll(text)) {
            val written = m.value.trim()
            val digits = written.count(Char::isDigit)
            val international = written.startsWith("+") || written.startsWith("00")
            if (digits > 15 || digits < (if (international) 8 else 9) || !free(m.range)) continue
            val number = written.filter(Char::isDigit).let { if (written.startsWith("+")) "+$it" else if (written.startsWith("00")) "+" + it.drop(2) else it }
            found += m.range to Detected(DetectedKind.PHONE, written, number)
        }
        return found.sortedBy { it.first.first }.distinctBy { it.second }
    }

    /** Only the entities. */
    fun entities(text: String): List<Detected> = find(text).map { it.second }

    /** ISO date for day/month/year strings (2-digit year = 20yy; a month over 12 = US order), null if impossible. */
    private fun date(d: String, m: String, y: String): String? {
        var day = d.toIntOrNull() ?: return null
        var month = m.toIntOrNull() ?: return null
        val year = (y.toIntOrNull() ?: return null).let { if (y.length == 2) 2000 + it else it }
        if (month > 12 && day <= 12) day = month.also { month = day }
        if (month !in 1..12 || day !in 1..31 || year !in 1900..2100) return null
        return String.format(java.util.Locale.ROOT, "%04d-%02d-%02d", year, month, day)
    }
}
