package pl.genor.glassvision

import androidx.camera.core.ImageProxy
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.Executor

/** Mode switching: one backend at a time, clean teardown, and no events from a replaced backend. */
class BackendSwitchTest {
    private class FakeBackend(override val mode: ProcessingMode, val listener: RecognitionBackend.Listener) : RecognitionBackend {
        val calls = mutableListOf<String>()
        override fun start() { calls += "start" }
        override fun stop() { calls += "stop" }
        override fun shutdown() { calls += "shutdown" }
        override fun applySettings(settings: StreamSettings) { calls += "settings" }
        override fun setPipelineSettings(settings: PipelineSettings) { calls += "pipelines" }
        override fun sendCommand(pipeline: String, action: String, args: Map<String, OptionValue>): Long? = null
        override fun canAcceptFrame() = true
        override fun submitFrame(image: ImageProxy, settings: StreamSettings) = true
    }

    private class Recorder : RecognitionBackend.Listener {
        val events = mutableListOf<String>()
        override fun onStateChanged(state: ConnectionState, detail: String?) { events += "state:$state" }
        override fun onHello(hello: ServerMessage.Hello) { events += "hello:${hello.device}" }
        override fun onPipelineStatus(change: ServerMessage.PipelineStatusChange) { events += "status:${change.pipeline}" }
        override fun onDetections(result: ServerMessage.Detections, receivedAtMs: Long) { events += "detections:${result.seq}" }
        override fun onCommandResult(result: ServerMessage.CommandResult) { events += "result:${result.req}" }
    }

    /** Main-thread stand-in: queues posts until [drain]. */
    private class QueueExecutor : Executor {
        val queue = ArrayDeque<Runnable>()
        override fun execute(command: Runnable) { queue += command }
        fun drain() { while (queue.isNotEmpty()) queue.removeFirst().run() }
    }

    private val recorder = Recorder()
    private val main = QueueExecutor()
    private val created = mutableListOf<FakeBackend>()
    private val switch = BackendSwitch(recorder, main) { mode, l -> FakeBackend(mode, l).also { created += it } }

    private fun detections(seq: Long) = ServerMessage.Detections(seq, 0, 1, 1, 0.0, 0.0, emptyMap(), emptyList(), emptyList())

    @Test
    fun select_configures_before_start_and_only_starts_when_running() {
        assertTrue(switch.select(ProcessingMode.SERVER) { it.applySettings(StreamSettings()) })
        assertEquals(listOf("settings"), created[0].calls) // not started yet
        switch.start()
        assertEquals(listOf("settings", "start"), created[0].calls)
        assertFalse("same mode: nothing happens", switch.select(ProcessingMode.SERVER))
        assertEquals(1, created.size)
    }

    @Test
    fun switching_shuts_the_old_backend_down_and_starts_the_new_one() {
        switch.select(ProcessingMode.SERVER)
        switch.start()
        assertTrue(switch.select(ProcessingMode.LOCAL) { it.setPipelineSettings(PipelineSettings()) })
        assertEquals(listOf("start", "shutdown"), created[0].calls)
        assertEquals(listOf("pipelines", "start"), created[1].calls)
        assertSame(created[1], switch.current)
        assertEquals(ProcessingMode.LOCAL, switch.current?.mode)
    }

    @Test
    fun events_of_a_replaced_backend_are_dropped_even_if_already_posted() {
        switch.select(ProcessingMode.SERVER)
        switch.start()
        val old = created[0].listener
        old.onDetections(detections(1), 0) // posted to the main thread, not delivered yet
        switch.select(ProcessingMode.LOCAL)
        old.onDetections(detections(2), 0) // after the switch
        created[1].listener.onStateChanged(ConnectionState.CONNECTED, "On this phone")
        main.drain()
        assertEquals(listOf("state:CONNECTED"), recorder.events)
    }

    @Test
    fun events_arrive_on_the_delivery_executor_in_order() {
        switch.select(ProcessingMode.LOCAL)
        val l = created[0].listener
        l.onStateChanged(ConnectionState.CONNECTED, null)
        l.onPipelineStatus(ServerMessage.PipelineStatusChange("pose", PipelineStatus.LOADING, null))
        l.onCommandResult(ServerMessage.CommandResult(7, "pose", "capture", true, "", emptyList(), null))
        assertTrue(recorder.events.isEmpty())
        main.drain()
        assertEquals(listOf("state:CONNECTED", "status:pose", "result:7"), recorder.events)
    }

    @Test
    fun stop_start_and_shutdown() {
        switch.select(ProcessingMode.SERVER)
        switch.start()
        switch.stop()
        switch.start()
        switch.shutdown()
        assertEquals(listOf("start", "stop", "start", "shutdown"), created[0].calls)
        assertNull(switch.current)
        created[0].listener.onStateChanged(ConnectionState.DISCONNECTED, null)
        main.drain()
        assertTrue(recorder.events.isEmpty())
    }
}
