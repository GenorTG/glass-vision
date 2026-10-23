package pl.genor.glassvision

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizer
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/*
 * The on-device pipelines: MediaPipe Tasks Vision 0.10.35 (objects, gestures, pose, faces, + hands for pose capture),
 * and text (OnDeviceText.kt). Android-only glue around the pure parts (LocalBackend, LocalTracker, PoseNames, OpenPose,
 * TextLayout), which are unit-tested on the JVM; these run in the instrumented tests.
 * API: https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/android (and the gesture_recognizer,
 * pose_landmarker, face_detector, hand_landmarker pages)
 */

/** A [LocalFrame] backed by an upright ARGB_8888 bitmap (shared read-only by the lanes; never recycled early). */
class BitmapFrame(
    val bitmap: Bitmap,
    override val capturedAtMs: Long = System.currentTimeMillis(),
    override val uptimeMs: Long = SystemClock.uptimeMillis(),
    override val prepMs: Double = 0.0,
) : LocalFrame {
    override val width: Int get() = bitmap.width
    override val height: Int get() = bitmap.height

    /** One MPImage per frame and pipeline run (cheap: wraps the bitmap). */
    fun mpImage(): MPImage = BitmapImageBuilder(bitmap).build()
}

/** CameraX RGBA frame → upright bitmap, long side ≤ the "Frame size" setting (no JPEG, no network). */
object CameraFrames : LocalFrameSource {
    override fun convert(image: ImageProxy, longSide: Int): LocalFrame {
        val capturedAt = System.currentTimeMillis()
        val uptime = SystemClock.uptimeMillis()
        val t0 = System.nanoTime()
        val source = image.toBitmap()
        val upright = FrameEncoder.uprightBitmap(source, image.imageInfo.rotationDegrees, longSide)
        if (upright !== source) source.recycle()
        return BitmapFrame(upright, capturedAt, uptime, (System.nanoTime() - t0) / 1e6)
    }
}

private fun LocalFrame.bitmapFrame(): BitmapFrame =
    this as? BitmapFrame ?: throw IllegalArgumentException("on-device pipelines need a BitmapFrame")

private fun Map<String, OptionValue>.num(key: String, default: Double): Double = (this[key] as? OptionValue.Num)?.value ?: default
private fun Map<String, OptionValue>.flag(key: String, default: Boolean): Boolean = (this[key] as? OptionValue.Bool)?.value ?: default

/**
 * Creates the on-device pipelines: downloads the model via [models] (first use), then the MediaPipe task in VIDEO
 * mode (one camera stream, so tracking between frames speeds up pose and hands). With [gpu] it tries MediaPipe's
 * GPU delegate first, including one warm-up run, and falls back to the CPU if either fails.
 */
class OnDeviceLoader(
    context: Context,
    private val models: ModelManager,
    private val captures: CaptureStore,
) : LocalPipelineLoader {
    private val context = context.applicationContext

    override fun configKey(pipeline: String, options: Map<String, OptionValue>, gpu: Boolean): String = when (pipeline) {
        LocalCatalog.GESTURES -> "gestures/${hands(options)}/${minConf(options)}/$gpu"
        LocalCatalog.TEXT -> "text/${options[LocalCatalog.OPT_OCR_LANGUAGE]}"
        else -> "$pipeline/${LocalCatalog.modelFor(pipeline, options)}/$gpu"
    }

    override fun load(pipeline: String, options: Map<String, OptionValue>, gpu: Boolean): LocalPipeline {
        if (pipeline == LocalCatalog.TEXT) return loadText(options)
        val model = LocalCatalog.modelFor(pipeline, options) ?: throw IllegalArgumentException("unknown pipeline $pipeline")
        val file = models.ensure(model.spec)
        val create: (Delegate) -> MediaPipePipeline = { delegate ->
            val base = BaseOptions.builder().setModelAssetPath(file.absolutePath).setDelegate(delegate).build()
            when (pipeline) {
                LocalCatalog.OBJECTS -> ObjectsPipeline(
                    ObjectDetector.createFromOptions(
                        context,
                        ObjectDetector.ObjectDetectorOptions.builder().setBaseOptions(base)
                            .setRunningMode(RunningMode.VIDEO)
                            // Detect down to the tracker floor; the user's threshold only gates new tracks (like the server).
                            .setScoreThreshold(LocalTracker.LOW_CONF)
                            .setMaxResults(MAX_OBJECTS)
                            .build(),
                    ),
                )
                LocalCatalog.GESTURES -> GesturesPipeline(
                    GestureRecognizer.createFromOptions(
                        context,
                        GestureRecognizer.GestureRecognizerOptions.builder().setBaseOptions(base)
                            .setRunningMode(RunningMode.VIDEO)
                            .setNumHands(hands(options))
                            .setMinHandDetectionConfidence(minConf(options))
                            .setMinHandPresenceConfidence(minConf(options))
                            .setMinTrackingConfidence(minConf(options))
                            .build(),
                    ),
                )
                LocalCatalog.POSE -> PosePipeline(
                    PoseLandmarker.createFromOptions(
                        context,
                        PoseLandmarker.PoseLandmarkerOptions.builder().setBaseOptions(base)
                            .setRunningMode(RunningMode.VIDEO)
                            .setNumPoses(MAX_POSES)
                            .build(),
                    ),
                    handLandmarker = { createHandLandmarker() },
                    captures = captures,
                )
                LocalCatalog.FACES -> FacesPipeline(
                    FaceDetector.createFromOptions(
                        context,
                        FaceDetector.FaceDetectorOptions.builder().setBaseOptions(base)
                            .setRunningMode(RunningMode.VIDEO)
                            .build(),
                    ),
                )
                else -> throw IllegalArgumentException("unknown pipeline $pipeline")
            }
        }
        if (gpu) {
            var onGpu: MediaPipePipeline? = null
            try {
                onGpu = create(Delegate.GPU)
                onGpu.warmUp()
                return onGpu
            } catch (e: Exception) {
                Log.w(TAG, "$pipeline: GPU delegate failed, using the CPU", e)
                try {
                    onGpu?.close()
                } catch (_: Exception) {
                    // Half-initialized: nothing more to release.
                }
            }
            return create(Delegate.CPU).also { it.note = "GPU unavailable — running on the CPU" }
        }
        return create(Delegate.CPU)
    }

    /**
     * ML Kit text (+ Tesseract on line crops, see [OcrPlan]). A chosen pack downloads now, so a failed download is
     * the pipeline's error; so does the fallback pack of a Play-services script that isn't there (yet).
     */
    private fun loadText(options: Map<String, OptionValue>): LocalPipeline {
        val plan = planFor(options, models)
        val mlKit = MlKitReaders(context)
        val tesseract = TesseractCrops(models)
        try {
            val available = mlKit.available(plan.script)
            val packs = if (available) plan.packs else plan.fallback + plan.packs
            if (!plan.auto) packs.forEach(tesseract::prepare)
            val note = if (available) null else "${plan.script.title} text model downloads through Google Play services; Tesseract reads meanwhile"
            return TextPipeline(TextEngine(mlKit, tesseract), plan, Translations(), note)
        } catch (e: Exception) {
            mlKit.close()
            tesseract.close()
            throw e
        }
    }

    override fun textReader(options: () -> Map<String, OptionValue>): TextReader = LocalTextReader(context, models, options)

    /** Pose capture's hands: IMAGE mode on the CPU (one-off, like the server), model downloaded on first capture. */
    private fun createHandLandmarker(): HandLandmarker {
        val file = models.ensure(LocalModel.HAND_LANDMARKER.spec)
        return HandLandmarker.createFromOptions(
            context,
            HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(file.absolutePath).setDelegate(Delegate.CPU).build())
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(MAX_CAPTURE_HANDS)
                .build(),
        )
    }

    private fun hands(o: Map<String, OptionValue>) = o.num(LocalCatalog.OPT_NUM_HANDS, 2.0).toInt().coerceIn(1, 4)

    private fun minConf(o: Map<String, OptionValue>) = Math.round(o.num(LocalCatalog.OPT_MIN_CONF, 0.5) * 1000).toFloat() / 1000f

    companion object {
        private const val TAG = "OnDeviceLoader"
        const val MAX_OBJECTS = 25
        /** Every person, like the server (VIDEO mode only re-detects when a pose is lost). */
        const val MAX_POSES = 4
        const val MAX_CAPTURE_HANDS = 4
    }
}

/** A MediaPipe task wrapped as a [LocalPipeline]; [warmUp] proves the delegate works before it goes live. */
abstract class MediaPipePipeline : LocalPipeline {
    override var note: String? = null

    fun warmUp() {
        val blank = Bitmap.createBitmap(WARMUP_W, WARMUP_H, Bitmap.Config.ARGB_8888)
        run(BitmapFrame(blank, 0, 0), 0, emptyMap())
    }

    private companion object {
        const val WARMUP_W = 320
        const val WARMUP_H = 240
    }
}

class ObjectsPipeline(private val detector: ObjectDetector) : MediaPipePipeline() {
    override fun run(frame: LocalFrame, timestampMs: Long, options: Map<String, OptionValue>): List<PixelDetection> {
        val f = frame.bitmapFrame()
        val result = detector.detectForVideo(f.mpImage(), timestampMs)
        return result.detections().mapNotNull { d ->
            val top = d.categories().maxByOrNull { it.score() } ?: return@mapNotNull null
            val b = d.boundingBox()
            box(b.left, b.top, b.right, b.bottom, f, top.score(), top.categoryName().ifEmpty { top.displayName() })
        }
    }

    override fun close() = detector.close()
}

class FacesPipeline(private val detector: FaceDetector) : MediaPipePipeline() {
    override fun run(frame: LocalFrame, timestampMs: Long, options: Map<String, OptionValue>): List<PixelDetection> {
        val f = frame.bitmapFrame()
        return detector.detectForVideo(f.mpImage(), timestampMs).detections().mapNotNull { d ->
            val score = d.categories().firstOrNull()?.score() ?: 1f
            val b = d.boundingBox()
            box(b.left, b.top, b.right, b.bottom, f, score, "face")
        }
    }

    override fun close() = detector.close()
}

private fun box(x1: Float, y1: Float, x2: Float, y2: Float, f: LocalFrame, conf: Float, label: String): PixelDetection? {
    val l = x1.coerceIn(0f, f.width.toFloat())
    val t = y1.coerceIn(0f, f.height.toFloat())
    val r = x2.coerceIn(0f, f.width.toFloat())
    val b = y2.coerceIn(0f, f.height.toFloat())
    if (r <= l || b <= t) return null
    return PixelDetection(l, t, r, b, conf, label)
}

/** Same output as the server's gestures pipeline: 21 points, padded box, gesture or "hand", handedness as detail. */
class GesturesPipeline(private val recognizer: GestureRecognizer) : MediaPipePipeline() {
    override fun run(frame: LocalFrame, timestampMs: Long, options: Map<String, OptionValue>): List<PixelDetection> {
        val f = frame.bitmapFrame()
        val result = recognizer.recognizeForVideo(f.mpImage(), timestampMs)
        val w = f.width.toFloat()
        val h = f.height.toFloat()
        return result.landmarks().mapIndexedNotNull { i, landmarks ->
            if (landmarks.isEmpty()) return@mapIndexedNotNull null
            val points = landmarks.map { KeyPoint(it.x() * w, it.y() * h, 1f) }
            val minX = points.minOf { it.x }
            val maxX = points.maxOf { it.x }
            val minY = points.minOf { it.y }
            val maxY = points.maxOf { it.y }
            val padX = (maxX - minX) * BlazePose.BOX_PAD
            val padY = (maxY - minY) * BlazePose.BOX_PAD
            val hand = result.handedness().getOrNull(i)?.firstOrNull()
            val gesture = result.gestures().getOrNull(i)?.firstOrNull()
            val (label, conf) = if (gesture != null && gesture.categoryName().isNotEmpty() && gesture.categoryName() != "None") {
                gesture.categoryName() to gesture.score()
            } else {
                "hand" to (hand?.score() ?: 1f)
            }
            // MediaPipe labels handedness for a mirrored selfie image; these frames are not mirrored.
            val detail = hand?.categoryName()?.let { UNMIRROR[it] ?: it }
            PixelDetection(
                (minX - padX).coerceAtLeast(0f), (minY - padY).coerceAtLeast(0f),
                (maxX + padX).coerceAtMost(w), (maxY + padY).coerceAtMost(h),
                conf, label, detail, points,
            )
        }
    }

    override fun close() = recognizer.close()

    private companion object {
        val UNMIRROR = mapOf("Left" to "Right", "Right" to "Left")
    }
}

/**
 * BlazePose (33 landmarks) mapped to COCO-17, so the overlay skeleton, [PoseNames] and the OpenPose export work as
 * with the server's YOLO pose. Action `capture` → the server's 5 capture files, rendered on the phone.
 */
class PosePipeline(
    private val landmarker: PoseLandmarker,
    private val handLandmarker: () -> HandLandmarker,
    private val captures: CaptureStore,
) : MediaPipePipeline() {
    private var hands: HandLandmarker? = null

    private class Body(val coco: List<KeyPoint>, val box: FloatArray)

    private fun bodies(f: BitmapFrame, timestampMs: Long): List<Body> =
        landmarker.detectForVideo(f.mpImage(), timestampMs).landmarks().mapNotNull { lms ->
            if (lms.size < BlazePose.LANDMARKS) return@mapNotNull null
            val norm = lms.map(::keyPoint)
            val box = BlazePose.box(norm, f.width, f.height) ?: return@mapNotNull null
            Body(BlazePose.toCoco(norm, f.width, f.height), box)
        }

    override fun run(frame: LocalFrame, timestampMs: Long, options: Map<String, OptionValue>): List<PixelDetection> {
        val names = options.flag(LocalCatalog.OPT_NAMES, true)
        return bodies(frame.bitmapFrame(), timestampMs).map { b ->
            val conf = b.coco.map { it.v }.average().toFloat()
            PixelDetection(
                b.box[0], b.box[1], b.box[2], b.box[3], conf,
                label = if (names) PoseNames.name(b.coco) else "person",
                detail = if (names) "person" else null,
                points = b.coco,
            )
        }
    }

    override fun prepare(action: String) {
        if (action == LocalCatalog.ACTION_CAPTURE && hands == null) hands = handLandmarker()
    }

    override fun command(
        action: String,
        args: Map<String, OptionValue>,
        frame: LocalFrame?,
        timestampMs: Long,
        options: Map<String, OptionValue>,
    ): CommandOutcome {
        if (action != LocalCatalog.ACTION_CAPTURE) return CommandOutcome(false, "Unknown action: $action")
        val f = frame?.bitmapFrame() ?: return CommandOutcome(false, "No frame yet: start the camera first")
        val w = f.width
        val h = f.height
        val bodies = bodies(f, timestampMs).filter { b -> b.coco.count { it.v >= OpenPose.MIN_CONF } >= MIN_CAPTURE_POINTS }
        if (bodies.isEmpty()) return CommandOutcome(false, "No person in view")
        val people = bodies.map { b -> b.coco.map { Triple(it.x.toDouble(), it.y.toDouble(), it.v.toDouble()) } }

        val found = (hands ?: handLandmarker().also { hands = it }).detect(f.mpImage()).landmarks()
            .filter { it.size == 21 }
            .map { hand -> hand.map { (it.x() * w).toDouble() to (it.y() * h).toDouble() } }
        val assignment = OpenPose.assignHands(people, found.map { it[0] }, bodies.map { b -> DoubleArray(4) { b.box[it].toDouble() } })
        fun norm(hand: List<Pair<Double, Double>>) = hand.map { (x, y) -> NormPoint(x / w, y / h) }
        fun handOf(person: Int, side: OpenPose.Side) = assignment.assigned[person to side]?.let { norm(found[it]) }
        val persons = people.mapIndexed { i, points ->
            OpenPosePerson(OpenPose.cocoToOpenPose(points, w, h), handOf(i, OpenPose.Side.LEFT), handOf(i, OpenPose.Side.RIGHT))
        }
        val loose = assignment.loose.map { norm(found[it]) }
        val files = captures.write(f.bitmap, persons, loose)
        val n = persons.size
        return CommandOutcome(true, "Captured $n ${if (n == 1) "person" else "people"} → ${files.size} files", files)
    }

    override fun close() {
        landmarker.close()
        hands?.close()
        hands = null
    }

    private companion object {
        /** A capture needs a body, not a stray hand: at least this many COCO points seen (server: person conf ≥ 0.4). */
        const val MIN_CAPTURE_POINTS = 5

        fun keyPoint(lm: NormalizedLandmark) = KeyPoint(lm.x(), lm.y(), lm.visibility().orElse(0f))
    }
}

/**
 * Pose captures made on the phone: the server's 5 files (`captures.CAPTURE_FILES`: photo.jpg, openpose_body.png,
 * openpose_full.png, pose_keypoints.json, pose_on_photo.jpg) written into [root]/<id>/ (app cache), returned as
 * `file:` result files that the normal file-saving path copies into Pictures/GlassVision/<id>/ and Downloads.
 */
class CaptureStore(val root: File) {
    fun write(photo: Bitmap, people: List<OpenPosePerson>, looseHands: List<List<NormPoint>>): List<ServerMessage.CommandResult.ResultFile> {
        prune()
        val (_, dir) = newCapture()
        val w = photo.width
        val h = photo.height
        val pixels = IntArray(w * h)
        photo.getPixels(pixels, 0, w, 0, 0, w, h)
        for (i in pixels.indices) pixels[i] = pixels[i] and 0xFFFFFF
        val background = RgbRaster(w, h, pixels)

        save(dir, "photo.jpg") { photo.compress(Bitmap.CompressFormat.JPEG, PHOTO_QUALITY, it) }
        savePng(dir, "openpose_body.png", OpenPose.drawPoses(people, h, w, drawHands = false))
        savePng(dir, "openpose_full.png", OpenPose.drawPoses(people, h, w, extraHands = looseHands))
        File(dir, "pose_keypoints.json").writeText(OpenPose.encodePoseJson(people, h, w))
        val overlay = OpenPose.drawPoses(people, h, w, extraHands = looseHands, background = background)
        save(dir, "pose_on_photo.jpg") { toBitmap(overlay).compress(Bitmap.CompressFormat.JPEG, OVERLAY_QUALITY, it) }

        return FILES.map { (name, kind) -> ServerMessage.CommandResult.ResultFile(name, kind, File(dir, name).toURI().toString()) }
    }

    private fun newCapture(): Pair<String, File> {
        if (!root.isDirectory) root.mkdirs()
        while (true) {
            val id = SimpleDateFormat("yyyyMMdd-HHmmss-SSS", Locale.US).format(Date())
            val dir = File(root, id)
            if (dir.mkdir()) return id to dir
            Thread.sleep(1) // two captures in the same millisecond
        }
    }

    /** Copies already in the gallery: drop local leftovers older than a day. */
    private fun prune() {
        val cutoff = System.currentTimeMillis() - KEEP_MS
        root.listFiles()?.filter { it.isDirectory && it.lastModified() < cutoff }?.forEach { it.deleteRecursively() }
    }

    private fun savePng(dir: File, name: String, raster: RgbRaster) =
        save(dir, name) { toBitmap(raster).compress(Bitmap.CompressFormat.PNG, 100, it) }

    private fun save(dir: File, name: String, write: (FileOutputStream) -> Boolean) {
        FileOutputStream(File(dir, name)).use { if (!write(it)) error("encoding $name failed") }
    }

    private fun toBitmap(raster: RgbRaster): Bitmap {
        val argb = IntArray(raster.pixels.size) { raster.pixels[it] or OPAQUE }
        return Bitmap.createBitmap(argb, raster.width, raster.height, Bitmap.Config.ARGB_8888)
    }

    companion object {
        /** Same names, order and kinds as the server's `captures.CAPTURE_FILES`. */
        val FILES = listOf(
            "photo.jpg" to "photo",
            "openpose_body.png" to "openpose",
            "openpose_full.png" to "openpose",
            "pose_keypoints.json" to "json",
            "pose_on_photo.jpg" to "overlay",
        )
        private const val PHOTO_QUALITY = 95
        private const val OVERLAY_QUALITY = 92
        private const val OPAQUE = -0x1000000
        private const val KEEP_MS = 24L * 60 * 60 * 1000
    }
}
