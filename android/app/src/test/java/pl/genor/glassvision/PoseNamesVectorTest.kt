package pl.genor.glassvision

import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The Kotlin [PoseNames] port against labels the server's `pose_names.py` produced for the same keypoints
 * (`src/test/gen/gen_test_vectors.py`): the server tests' synthetic poses, the same with hidden points, and random
 * skeletons through every branch. Every label must match exactly.
 */
class PoseNamesVectorTest {
    private val vectors: JSONArray by lazy {
        JSONArray(requireNotNull(javaClass.classLoader?.getResource("vectors/pose_names.json")).readText())
    }

    private fun points(arr: JSONArray): List<Triple<Double, Double, Double>> = List(arr.length()) { i ->
        val p = arr.getJSONArray(i)
        Triple(p.getDouble(0), p.getDouble(1), p.getDouble(2))
    }

    @Test
    fun every_python_label_is_reproduced() {
        val mismatches = mutableListOf<String>()
        for (i in 0 until vectors.length()) {
            val v = vectors.getJSONObject(i)
            val got = PoseNames.name(points(v.getJSONArray("points")))
            val want = v.getString("label")
            if (got != want) mismatches += "#$i: want '$want', got '$got'"
        }
        assertTrue("${mismatches.size} of ${vectors.length()} differ:\n" + mismatches.take(20).joinToString("\n"), mismatches.isEmpty())
    }

    @Test
    fun vectors_cover_every_label_family() {
        val labels = (0 until vectors.length()).map { vectors.getJSONObject(it).getString("label") }.toSet()
        val parts = labels.flatMap { it.split(PoseNames.SEP) }.toSet()
        for (expected in listOf(
            "Standing", "Sitting", "Squatting", "Kneeling", "Lying down", "Bending over", "Upper body", "person",
            "Arms up", "T-pose", "Arms crossed", "Hands on hips", "Hand on head", "Pointing left", "Pointing right",
            "One arm up",
        )) {
            assertTrue("no vector for '$expected'", expected in parts)
        }
    }

    @Test
    fun keypoint_overload_matches() {
        val v = vectors.getJSONObject(1)
        val pts = points(v.getJSONArray("points"))
        assertEquals(v.getString("label"), PoseNames.name(pts.map { (x, y, c) -> KeyPoint(x.toFloat(), y.toFloat(), c.toFloat()) }))
    }

    @Test
    fun degenerate_input_is_person() {
        assertEquals("person", PoseNames.name(emptyList<Triple<Double, Double, Double>>()))
        assertEquals("person", PoseNames.name(List(17) { Triple(Double.NaN, 1.0, 1.0) }))
    }
}
