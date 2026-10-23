plugins {
    id("com.android.application") version "8.2.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.22" apply false
    // Snapshot tests of the real layouts (no emulator here). 1.3.3 = the Kotlin 1.9.22 / AGP 8.2 line.
    id("app.cash.paparazzi") version "1.3.3" apply false
}
