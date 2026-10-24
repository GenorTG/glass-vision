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

object ReadBlocks {
    /**
     * Blocks of a reader that only numbers lines (v3.2 server: `block` per line, translations per line): lines grouped
     * by `block` in order of their first line, box = union of the lines, kind guessed like the phone does
     * ([TextBlocks.kind]); a block's translation = its lines' translations (untranslated lines keep the original),
     * null when none of its lines was translated.
     */
    fun fromLines(lines: List<ReadLine>, translations: Map<String, List<String?>> = emptyMap()): List<ReadBlock> {
        val groups = LinkedHashMap<Int, MutableList<Int>>()
        lines.forEachIndexed { i, line -> groups.getOrPut(line.block) { ArrayList() } += i }
        return groups.map { (id, idx) ->
            val members = idx.map { lines[it] }
            val box = members.map { it.box }.reduce(TextBox::union)
            val kind = TextBlocks.kind(members.map { it.text }, members.map { it.box.width }, box.width)
            val texts = members.map { it.text }
            ReadBlock(
                id = id,
                lines = idx,
                box = box,
                quad = members.singleOrNull()?.quad,
                text = texts.joinToString("\n"),
                kind = kind,
                lang = majority(members.mapNotNull { it.lang }),
                suspect = TextSuspect.block(members.map { it.text to it.suspect }),
                translations = translations.mapValues { (_, t) ->
                    val entries = idx.map { t.getOrNull(it) }
                    if (entries.all { it == null }) {
                        null
                    } else {
                        val joined = entries.mapIndexed { i, e -> e ?: texts[i] }
                        if (kind == BlockKind.PROSE) TextBlocks.joinProse(joined) else joined.joinToString("\n")
                    }
                },
            )
        }
    }

    fun majority(codes: List<String>): String? = codes.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key
}

object ReadTextFormat {
    /** The recognized text: one block per paragraph ([ReadBlock.display]), blocks separated by a blank line. */
    fun original(reading: TextReading): String = reading.blocks.joinToString("\n\n") { it.display }

    /** [translation] (aligned with the blocks) the same way; untranslated blocks show the original. */
    fun translated(reading: TextReading, translation: List<String?>): String =
        reading.blocks.mapIndexed { i, b -> translation.getOrNull(i) ?: b.display }.joinToString("\n\n")

    /** True when no block got a translation (already in that language, digits only…). */
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

    /** One block for Copy/Share: its text, then each of its [translations] (language name → text) that exists. */
    fun block(block: ReadBlock, translations: List<Pair<String, String?>>): String = buildString {
        append(block.display)
        for ((name, t) in translations) {
            if (t == null) continue
            append("\n\n— ").append(name).append(" —\n").append(t)
        }
    }

    /** The most frequent detected source language (per line), or null if none was detected. */
    fun sourceLanguage(reading: TextReading): String? =
        ReadBlocks.majority(reading.lines.mapNotNull { it.lang }) ?: ReadBlocks.majority(reading.blocks.mapNotNull { it.lang })
}

/** How the Read text sheet shows its result. */
enum class ReadView {
    /** One card per block under a thumbnail. */
    CARDS,

    /** The still, large, with each block's translation painted over it (and the cards under it). */
    OVERLAY,
}

/** The remembered [ReadView]; [load]/[save] = the raw stored value. */
class ReadViewChoice(private val load: () -> String?, private val save: (String) -> Unit) {
    fun get(): ReadView = ReadView.entries.firstOrNull { it.name == load() } ?: ReadView.CARDS

    fun set(view: ReadView) = save(view.name)

    companion object {
        private const val KEY_VIEW = "view"

        fun of(context: Context): ReadViewChoice {
            val prefs = context.applicationContext.getSharedPreferences(ReadTargets.PREFS_NAME, Context.MODE_PRIVATE)
            return ReadViewChoice({ prefs.getString(KEY_VIEW, null) }, { prefs.edit().putString(KEY_VIEW, it).apply() })
        }
    }
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
        const val PREFS_NAME = "glass_vision_read_text"
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
