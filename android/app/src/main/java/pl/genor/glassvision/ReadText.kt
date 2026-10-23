package pl.genor.glassvision

import android.content.Context
import java.util.Locale

/*
 * Read text: the pure parts (JVM-testable) — when the button shows, how lines become readable blocks, what is copied
 * or shared, which languages to translate into.
 */

object ReadAvailability {
    const val TEXT_PIPELINE = "text"

    /** The Read button: connected/running, the text pipeline enabled, and a backend that can read stills. */
    fun canRead(
        connection: ConnectionState,
        pipelines: List<PipelineSpec>?,
        settings: PipelineSettings,
        hasReader: Boolean,
    ): Boolean {
        if (!hasReader || connection != ConnectionState.CONNECTED) return false
        val text = pipelines?.firstOrNull { it.id == TEXT_PIPELINE } ?: return false
        return settings.isEnabled(text)
    }
}

/** One sign/paragraph: [lines] are indices into [TextReading.lines], in reading order. */
data class ReadBlock(val id: Int, val lines: List<Int>)

object ReadBlocks {
    /** Lines grouped by `block`, blocks in order of their first line (lines are already in reading order). */
    fun of(lines: List<ReadLine>): List<ReadBlock> {
        val groups = LinkedHashMap<Int, MutableList<Int>>()
        lines.forEachIndexed { i, line -> groups.getOrPut(line.block) { ArrayList() } += i }
        return groups.map { (id, idx) -> ReadBlock(id, idx) }
    }
}

object ReadTextFormat {
    /** The recognized text: lines of a block on separate lines, blocks separated by a blank line. */
    fun original(reading: TextReading): String = render(reading) { i -> reading.lines[i].text }

    /** [translation] (aligned with the lines) the same way; untranslated entries show the original line. */
    fun translated(reading: TextReading, translation: List<String?>): String =
        render(reading) { i -> translation.getOrNull(i) ?: reading.lines[i].text }

    /** True when no line got a translation (already in that language, digits only…). */
    fun isUnchanged(translation: List<String?>): Boolean = translation.all { it == null }

    /** Text for Share: the original, then each translation under its language name. */
    fun share(reading: TextReading, translations: List<Pair<String, List<String?>>>): String = buildString {
        append(original(reading))
        for ((name, t) in translations) {
            if (isUnchanged(t)) continue
            append("\n\n— ").append(name).append(" —\n")
            append(translated(reading, t))
        }
    }

    /** The most frequent detected source language, or null if none was detected. */
    fun sourceLanguage(reading: TextReading): String? =
        reading.lines.mapNotNull { it.lang }.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key

    private fun render(reading: TextReading, text: (Int) -> String): String =
        ReadBlocks.of(reading.lines).joinToString("\n\n") { b -> b.lines.joinToString("\n") { text(it) } }
}

/**
 * The target languages Read text translates into, remembered across reads (codes, in the order the user added them).
 * [load]/[save] = the raw stored value; null = never chosen yet.
 */
class ReadTargets(private val load: () -> String?, private val save: (String) -> Unit) {

    /** Stored targets, or null if the user never chose any. */
    fun stored(): List<String>? = decode(load())

    fun set(codes: List<String>) = save(encode(codes))

    fun add(code: String): List<String> = (current() + code).distinct().also(::set)

    fun remove(code: String): List<String> = (current() - code).also(::set)

    /** What to translate a new read into: the stored choice, or the phone's language; only codes [offered] has. */
    fun forRead(offered: List<TranslationLanguage>, phoneLanguage: String = Locale.getDefault().language): List<String> =
        initial(stored(), phoneLanguage, offered)

    private fun current(): List<String> = stored().orEmpty()

    companion object {
        private const val PREFS_NAME = "glass_vision_read_text"
        private const val KEY_TARGETS = "translate_to"

        fun of(context: Context): ReadTargets {
            val prefs = context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            return ReadTargets({ prefs.getString(KEY_TARGETS, null) }, { prefs.edit().putString(KEY_TARGETS, it).apply() })
        }

        fun encode(codes: List<String>): String = codes.map(String::trim).filter(String::isNotEmpty).distinct().joinToString(",")

        fun decode(raw: String?): List<String>? =
            raw?.split(',')?.map(String::trim)?.filter(String::isNotEmpty)?.distinct()

        /**
         * [stored] if the user chose before (an empty choice stays empty), else [phoneLanguage]. With a known
         * [offered] list, codes the reader doesn't offer are dropped; an empty [offered] (not loaded) keeps them.
         */
        fun initial(stored: List<String>?, phoneLanguage: String, offered: List<TranslationLanguage>): List<String> {
            val wanted = stored ?: listOf(phoneLanguage).filter { it.isNotBlank() }
            if (offered.isEmpty()) return wanted
            val codes = offered.mapTo(HashSet()) { it.code }
            return wanted.filter { it in codes }
        }
    }
}

/** Display name of a language code: the reader's own name, else the phone's, else the code. */
fun languageName(code: String, offered: List<TranslationLanguage>): String =
    offered.firstOrNull { it.code == code }?.name
        ?: Locale(code).getDisplayLanguage(Locale.getDefault()).takeIf { it.isNotBlank() && it != code }
            ?.replaceFirstChar { it.titlecase(Locale.getDefault()) }
        ?: code
