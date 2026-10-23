package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Parses messages captured from the real v3 server (the `_v3.json` files in src/test/resources, recorded with the live
 * glass-vision service on zidane.jpg with objects + faces + pose on). Guards against the two sides drifting.
 */
class ServerFixtureTest {
    private fun fixture(name: String): String =
        requireNotNull(javaClass.classLoader?.getResource(name)) { "missing fixture $name" }.readText()

    @Test
    fun real_hello_describes_all_pipelines() {
        val hello = Protocol.parse(fixture("hello_v3.json")) as ServerMessage.Hello
        assertEquals(3, hello.protocol)
        val byId = hello.pipelines.associateBy { it.id }
        assertEquals(listOf("objects", "open_vocab", "faces", "text", "gestures", "pose"), hello.pipelines.map { it.id })
        assertTrue(byId.getValue("objects").defaultEnabled)
        assertEquals(OptionType.TEXT, byId.getValue("open_vocab").options.single().type)
        assertEquals(listOf("enroll", "list"), byId.getValue("faces").actions.map { it.id })
        assertEquals("name", byId.getValue("faces").actions.first().args.single().key)
        assertEquals(400, byId.getValue("text").intervalMs)
        assertEquals(OptionType.CHOICE, byId.getValue("text").options.first { it.key == "languages" }.type)
        assertEquals(21, byId.getValue("gestures").skeleton.flatMap { listOf(it.first, it.second) }.toSet().size)
        assertEquals(17, byId.getValue("pose").skeleton.flatMap { listOf(it.first, it.second) }.toSet().size)
    }

    @Test
    fun real_detections_carry_tracks_of_three_pipelines() {
        val msg = Protocol.parse(fixture("detections_v3.json")) as ServerMessage.Detections
        assertEquals(setOf("objects", "faces", "pose"), msg.tracks.map { it.pipeline }.toSet())
        assertEquals(setOf("objects", "faces", "pose"), msg.pipelineMs.keys)
        for (t in msg.tracks) {
            assertTrue(t.trackId!!.startsWith(t.pipeline + ":"))
            assertTrue(t.box.x1 in 0f..1f && t.box.x2 in 0f..1f && t.box.x1 < t.box.x2)
        }
        assertTrue(msg.tracks.filter { it.pipeline == "pose" }.all { it.points.size == 17 })
        assertTrue(msg.tracks.filter { it.pipeline == "faces" }.all { it.points.size == 5 })
    }

    @Test
    fun real_command_result_parses() {
        val msg = Protocol.parse(fixture("command_result_v3.json")) as ServerMessage.CommandResult
        assertEquals(7L, msg.req)
        assertEquals("faces", msg.pipeline)
        assertTrue(msg.ok)
    }
}
