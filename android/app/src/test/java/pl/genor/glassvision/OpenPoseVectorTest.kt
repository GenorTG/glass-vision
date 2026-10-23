package pl.genor.glassvision

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.zip.GZIPInputStream

/**
 * The OpenPose export port ([OpenPose], [CvDraw]) against the server's `openpose.py` / `pose.assign_hands` output
 * (`src/test/gen/gen_test_vectors.py`): COCO→OpenPose mapping, hand assignment, the exact pose_keypoints.json text,
 * and the rendered images — colour samples at every limb midpoint and joint, plus the whole PNG pixel by pixel.
 */
class OpenPoseVectorTest {
    private val vectors: JSONObject by lazy {
        JSONObject(requireNotNull(javaClass.classLoader?.getResource("vectors/openpose.json")).readText())
    }
    private val scene get() = vectors.getJSONObject("scene")

    private fun triples(arr: JSONArray) = List(arr.length()) { i ->
        val p = arr.getJSONArray(i)
        Triple(p.getDouble(0), p.getDouble(1), p.getDouble(2))
    }

    private fun normPoints(arr: JSONArray?): List<NormPoint>? = arr?.let { a ->
        List(a.length()) { i -> NormPoint(a.getJSONArray(i).getDouble(0), a.getJSONArray(i).getDouble(1)) }
    }

    @Test
    fun coco_to_openpose_matches_python() {
        val cases = vectors.getJSONArray("coco_to_openpose")
        for (c in 0 until cases.length()) {
            val v = cases.getJSONObject(c)
            val got = OpenPose.cocoToOpenPose(triples(v.getJSONArray("points")), v.getInt("w"), v.getInt("h"))
            val want = v.getJSONArray("body")
            assertEquals(18, got.size)
            for (i in 0 until 18) {
                if (want.isNull(i)) {
                    assertNull("case $c point $i", got[i])
                } else {
                    val p = want.getJSONArray(i)
                    assertEquals("case $c point $i", NormPoint(p.getDouble(0), p.getDouble(1)), got[i])
                }
            }
        }
    }

    @Test
    fun assign_hands_matches_python() {
        val cases = vectors.getJSONArray("assign_hands")
        for (c in 0 until cases.length()) {
            val v = cases.getJSONObject(c)
            val peopleArr = v.getJSONArray("people")
            val people = List(peopleArr.length()) { triples(peopleArr.getJSONArray(it)) }
            val boxesArr = v.getJSONArray("boxes")
            val boxes = List(boxesArr.length()) { i -> DoubleArray(4) { boxesArr.getJSONArray(i).getDouble(it) } }
            val wristsArr = v.getJSONArray("hand_wrists")
            val wrists = List(wristsArr.length()) { wristsArr.getJSONArray(it).getDouble(0) to wristsArr.getJSONArray(it).getDouble(1) }

            val got = OpenPose.assignHands(people, wrists, boxes)
            val want = v.getJSONArray("assigned")
            val wantMap = (0 until want.length()).associate { i ->
                val a = want.getJSONArray(i)
                val side = if (a.getString(1) == "left") OpenPose.Side.LEFT else OpenPose.Side.RIGHT
                (a.getInt(0) to side) to a.getInt(2)
            }
            assertEquals("case $c", wantMap, got.assigned)
            val loose = v.getJSONArray("loose")
            assertEquals("case $c loose", List(loose.length()) { loose.getInt(it) }, got.loose)
        }
    }

    private fun scenePeople(): Pair<List<OpenPosePerson>, List<List<NormPoint>>> {
        val w = scene.getInt("w")
        val h = scene.getInt("h")
        val arr = scene.getJSONArray("people")
        val people = List(arr.length()) { i ->
            val p = arr.getJSONObject(i)
            OpenPosePerson(
                OpenPose.cocoToOpenPose(triples(p.getJSONArray("coco")), w, h),
                normPoints(p.optJSONArray("left_hand")),
                normPoints(p.optJSONArray("right_hand")),
            )
        }
        val looseArr = scene.getJSONArray("loose_hands")
        val loose = List(looseArr.length()) { normPoints(looseArr.getJSONArray(it))!! }
        return people to loose
    }

    @Test
    fun pose_keypoints_json_is_byte_identical() {
        val (people, _) = scenePeople()
        assertEquals(scene.getString("json_text"), OpenPose.encodePoseJson(people, scene.getInt("h"), scene.getInt("w")))
    }

    private fun assertSamples(raster: RgbRaster, samples: JSONArray) {
        assertTrue(samples.length() > 20)
        for (i in 0 until samples.length()) {
            val s = samples.getJSONObject(i)
            val rgb = s.getJSONArray("rgb")
            val want = RgbRaster.rgb(rgb.getInt(0), rgb.getInt(1), rgb.getInt(2))
            val got = raster[s.getInt("x"), s.getInt("y")]
            assertEquals("${s.getString("kind")} at (${s.getInt("x")}, ${s.getInt("y")})", hex(want), hex(got))
        }
    }

    private fun hex(c: Int) = String.format("#%06X", c and 0xFFFFFF)

    /** Fraction of pixels identical to the Python render (its RGB canvas, raw rows, gzipped). */
    private fun agreement(raster: RgbRaster, name: String): Double {
        val rgb = GZIPInputStream(requireNotNull(javaClass.classLoader?.getResourceAsStream("vectors/$name"))).use { it.readBytes() }
        assertEquals(raster.width * raster.height * 3, rgb.size)
        var same = 0
        for (i in raster.pixels.indices) {
            val python = RgbRaster.rgb(rgb[3 * i].toInt(), rgb[3 * i + 1].toInt(), rgb[3 * i + 2].toInt())
            if (python == raster.pixels[i]) same++
        }
        return same.toDouble() / raster.pixels.size
    }

    @Test
    fun body_render_matches_python_at_limbs_joints_and_everywhere() {
        val (people, _) = scenePeople()
        val raster = OpenPose.drawPoses(people, scene.getInt("h"), scene.getInt("w"), drawHands = false)
        assertSamples(raster, scene.getJSONArray("samples_body"))
        assertEquals(1.0, agreement(raster, "openpose_body.rgb.gz"), 0.0)
    }

    @Test
    fun full_render_with_hands_matches_python() {
        val (people, loose) = scenePeople()
        val raster = OpenPose.drawPoses(people, scene.getInt("h"), scene.getInt("w"), extraHands = loose)
        assertSamples(raster, scene.getJSONArray("samples_full"))
        assertEquals(1.0, agreement(raster, "openpose_full.rgb.gz"), 0.0)
    }

    @Test
    fun background_is_copied_not_modified() {
        val photo = RgbRaster(40, 30, IntArray(40 * 30) { 0x323232 })
        val body = MutableList<NormPoint?>(18) { null }
        body[1] = NormPoint(0.25, 0.5)
        body[2] = NormPoint(0.75, 0.5)
        val out = OpenPose.drawPoses(listOf(OpenPosePerson(body)), 30, 40, background = photo)
        assertTrue(photo.pixels.all { it == 0x323232 })
        assertEquals(RgbRaster.rgb(153, 0, 0), out[20, 15]) // neck → right shoulder: COLORS[0] * 0.6
        assertEquals(0x323232, out[2, 2])
    }

    @Test
    fun hand_edge_colors_are_hsv_steps_rounded_like_cv2() {
        // What cv2.line paints for hand_edge_color(0..3) (checked against the server's cv2):
        // (255, 76.50000000000001, 0) → 77, (255, 229.49999999999997, 0) → 229.
        assertEquals(RgbRaster.rgb(255, 0, 0), OpenPose.handEdgeColor(0))
        assertEquals(RgbRaster.rgb(255, 77, 0), OpenPose.handEdgeColor(1))
        assertEquals(RgbRaster.rgb(255, 153, 0), OpenPose.handEdgeColor(2))
        assertEquals(RgbRaster.rgb(255, 229, 0), OpenPose.handEdgeColor(3))
    }
}
