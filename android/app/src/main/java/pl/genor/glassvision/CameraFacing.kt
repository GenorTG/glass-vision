package pl.genor.glassvision

/**
 * Which camera streams. CameraX's PreviewView shows the front camera mirrored (selfie view) while the
 * ImageAnalysis frames we send are not, so overlay positions must be mirrored to match ([isMirrored]).
 */
enum class CameraFacing(val prefValue: String) {
    BACK("back"), FRONT("front");

    val isMirrored: Boolean get() = this == FRONT

    fun flipped(): CameraFacing = if (this == BACK) FRONT else BACK

    companion object {
        fun fromPref(value: String?): CameraFacing = entries.firstOrNull { it.prefValue == value } ?: BACK

        /** [wanted] if the device has it, else the camera it does have (null = none at all). */
        fun resolve(wanted: CameraFacing, available: Set<CameraFacing>): CameraFacing? = when {
            wanted in available -> wanted
            else -> available.firstOrNull()
        }
    }
}
