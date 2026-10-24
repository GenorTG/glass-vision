package pl.genor.glassvision

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.Looper
import kotlin.math.sqrt

/**
 * "Hold still" gate for the Read shutter: open once the phone's angular speed stayed below [thresholdRadS] for
 * [holdMs] (the tap itself shakes the phone), or after [capMs] whatever happens, so a shaky hand still gets a picture.
 * Pure state machine over gyroscope samples; time in ms from any monotonic clock.
 */
class StillnessGate(
    private val startMs: Long,
    private val thresholdRadS: Float = THRESHOLD_RAD_S,
    private val holdMs: Long = HOLD_MS,
    private val capMs: Long = CAP_MS,
) {
    private var calmSinceMs: Long? = null
    var open = false
        private set

    /** One gyroscope sample (rad/s around x, y, z); true once the gate is open. */
    fun onSample(nowMs: Long, x: Float, y: Float, z: Float): Boolean {
        if (open) return true
        val speed = sqrt(x * x + y * y + z * z)
        if (speed < thresholdRadS) {
            val since = calmSinceMs ?: nowMs.also { calmSinceMs = it }
            if (nowMs - since >= holdMs) open = true
        } else {
            calmSinceMs = null
        }
        return open || expired(nowMs)
    }

    /** The wait is over without the phone calming down. */
    fun expired(nowMs: Long): Boolean = nowMs - startMs >= capMs

    companion object {
        /** ~8.6°/s: a hand-held phone at rest; walking or the tap's jolt is well above. */
        const val THRESHOLD_RAD_S = 0.15f
        const val HOLD_MS = 100L
        const val CAP_MS = 600L
    }
}

/** [StillnessGate] on the real gyroscope; without one (some cheap phones) it opens at once. Main thread. */
class HoldStill(context: Context) {
    private val sensors = context.getSystemService(SensorManager::class.java)
    private val gyro: Sensor? = sensors?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val main = Handler(Looper.getMainLooper())

    /** Calls [onStill] once, on the main thread, when the phone is still or after [capMs]. */
    fun await(capMs: Long = StillnessGate.CAP_MS, onStill: (calm: Boolean) -> Unit) {
        val manager = sensors
        if (gyro == null || manager == null) {
            main.post { onStill(false) }
            return
        }
        // The cap is the handler timeout below: sensor timestamps aren't guaranteed to share a clock with ours.
        val gate = StillnessGate(0L, capMs = Long.MAX_VALUE)
        var done = false
        lateinit var listener: SensorEventListener
        val timeout = Runnable { finish(manager, listener, done) { done = true; onStill(false) } }
        listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                if (gate.onSample(event.timestamp / 1_000_000, event.values[0], event.values[1], event.values[2])) {
                    main.removeCallbacks(timeout)
                    finish(manager, this, done) { done = true; onStill(gate.open) }
                }
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
        }
        manager.registerListener(listener, gyro, SensorManager.SENSOR_DELAY_GAME, main)
        main.postDelayed(timeout, capMs)
    }

    private fun finish(manager: SensorManager, listener: SensorEventListener, done: Boolean, then: () -> Unit) {
        if (done) return
        manager.unregisterListener(listener)
        then()
    }
}
