package pl.genor.glassvision

import org.junit.Assert.assertEquals
import org.junit.Test

/** [PyJson.float] against Python's `repr(float)` (the values were printed by the server's Python). */
class PyJsonTest {
    @Test
    fun matches_python_repr() {
        val cases = listOf(
            0.0 to "0.0", 1.0 to "1.0", -1.0 to "-1.0", 0.2 to "0.2", 0.1 to "0.1", 1e-4 to "0.0001",
            1e-5 to "1e-05", 0.00012345 to "0.00012345", 123.45 to "123.45", 100.0 to "100.0", 1e16 to "1e+16",
            1.5e-5 to "1.5e-05", 0.5216666666666666 to "0.5216666666666666", 2.5e-7 to "2.5e-07",
            1.0 / 3 to "0.3333333333333333", 12345678901234567.0 to "1.2345678901234568e+16",
            9999999999999998.0 to "9999999999999998.0", 0.1 + 0.2 to "0.30000000000000004", -0.0 to "-0.0",
        )
        for ((v, want) in cases) assertEquals("repr($v)", want, PyJson.float(v))
    }
}
