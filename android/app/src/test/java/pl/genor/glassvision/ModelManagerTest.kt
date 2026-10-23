package pl.genor.glassvision

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayInputStream
import java.io.IOException
import java.io.InputStream
import java.net.UnknownHostException
import java.security.MessageDigest
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/** [ModelManager]'s download state machine with a fake [Downloader]: verify, resume, failures, delete. */
class ModelManagerTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private val content = ByteArray(300_000) { (it * 31 % 251).toByte() }
    private val spec = ModelSpec("TEST", "test.task", "Test model", "https://example.invalid/test.task", content.size.toLong(), sha(content))

    private fun sha(b: ByteArray) = MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }

    /** Serves [content] (from the requested offset when [ranges]); [cutAt] = drop the connection after that byte. */
    private class FakeDownloader(
        var content: ByteArray,
        var ranges: Boolean = true,
        var cutAt: Int? = null,
        var error: IOException? = null,
    ) : Downloader {
        val offsets = mutableListOf<Long>()
        val calls = AtomicInteger()

        override fun open(url: String, offset: Long): Downloader.Body {
            calls.incrementAndGet()
            offsets += offset
            error?.let { throw it }
            val start = if (ranges) offset.toInt() else 0
            val end = cutAt?.coerceAtLeast(start) ?: content.size
            val bytes = content.copyOfRange(start, end)
            val cut = cutAt != null
            val stream: InputStream = object : ByteArrayInputStream(bytes) {
                override fun read(b: ByteArray, off: Int, len: Int): Int {
                    val n = super.read(b, off, len)
                    if (n < 0 && cut) throw IOException("Connection reset")
                    return n
                }
            }
            return Downloader.Body(stream, resumed = ranges && offset > 0)
        }
    }

    private fun manager(d: Downloader, free: Long = Long.MAX_VALUE) =
        ModelManager(tmp.root.resolve("models"), d, listOf(spec)) { free }

    @Test
    fun downloads_verifies_and_reuses() {
        val d = FakeDownloader(content)
        val m = manager(d)
        val progress = mutableListOf<Long>()
        val file = m.ensure(spec) { done, total -> assertEquals(spec.bytes, total); progress += done }
        assertArrayEquals(content, file.readBytes())
        assertTrue(m.isReady(spec))
        assertFalse(tmp.root.resolve("models/test.task.part").exists())
        assertEquals(spec.bytes, progress.last())
        m.ensure(spec)
        assertEquals("a ready model is not downloaded again", 1, d.calls.get())
        assertEquals(listOf(StoredModel("TEST", "Test model", spec.bytes)), m.downloaded())
    }

    @Test
    fun interrupted_download_resumes_from_the_part_file() {
        val d = FakeDownloader(content, cutAt = 100_000)
        val m = manager(d)
        val e = expectUnavailable { m.ensure(spec) }
        assertTrue(e.message!!, "Connection reset" in e.message!!)
        assertEquals(100_000L, tmp.root.resolve("models/test.task.part").length())
        assertFalse(m.isReady(spec))

        d.cutAt = null
        assertArrayEquals(content, m.ensure(spec).readBytes())
        assertEquals(listOf(0L, 100_000L), d.offsets)
    }

    @Test
    fun server_without_ranges_restarts_cleanly() {
        val d = FakeDownloader(content, cutAt = 50_000)
        val m = manager(d)
        expectUnavailable { m.ensure(spec) }
        d.cutAt = null
        d.ranges = false // answers 200 with the whole file
        assertArrayEquals(content, m.ensure(spec).readBytes())
    }

    @Test
    fun checksum_mismatch_is_rejected_and_discarded() {
        val bad = content.copyOf().also { it[1234] = (it[1234] + 1).toByte() }
        val m = manager(FakeDownloader(bad))
        val e = expectUnavailable { m.ensure(spec) }
        assertEquals("Model download corrupted (checksum mismatch) — try again", e.message)
        assertFalse(tmp.root.resolve("models/test.task.part").exists())
        assertFalse(m.isReady(spec))
    }

    @Test
    fun offline_says_so() {
        val m = manager(FakeDownloader(content, error = UnknownHostException("storage.googleapis.com")))
        assertEquals("No internet — model download failed", expectUnavailable { m.ensure(spec) }.message)
        assertTrue(m.downloaded().isEmpty())
    }

    @Test
    fun http_errors_and_full_disks_are_readable() {
        assertEquals("Model download failed (HTTP 404)", expectUnavailable { manager(FakeDownloader(content, error = HttpStatusException(404))).ensure(spec) }.message)
        val full = expectUnavailable { manager(FakeDownloader(content), free = 1000).ensure(spec) }
        assertTrue(full.message!!, full.message!!.startsWith("Not enough free space: Test model needs 0.3 MB"))
    }

    @Test
    fun delete_removes_the_model_and_its_part() {
        val d = FakeDownloader(content)
        val m = manager(d)
        m.ensure(spec)
        assertTrue(m.delete("TEST"))
        assertFalse(m.isReady(spec))
        assertTrue(m.downloaded().isEmpty())
        assertFalse(m.delete("TEST"))
        assertFalse(m.delete("UNKNOWN"))
        m.ensure(spec)
        assertEquals("downloaded again after delete", 2, d.calls.get())
    }

    @Test
    fun concurrent_callers_share_one_download() {
        val d = FakeDownloader(content)
        val m = manager(d)
        val pool = Executors.newFixedThreadPool(4)
        val start = CountDownLatch(1)
        val results = (1..4).map { pool.submit<ByteArray> { start.await(); m.ensure(spec).readBytes() } }
        start.countDown()
        results.forEach { assertArrayEquals(content, it.get(10, TimeUnit.SECONDS)) }
        pool.shutdown()
        assertEquals(1, d.calls.get())
    }

    private fun expectUnavailable(block: () -> Unit): ModelUnavailableException {
        try {
            block()
        } catch (e: ModelUnavailableException) {
            return e
        }
        fail("expected ModelUnavailableException")
        throw AssertionError()
    }
}
