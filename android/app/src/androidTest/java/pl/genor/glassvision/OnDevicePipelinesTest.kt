package pl.genor.glassvision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Every on-device pipeline on real images, on the CPU delegate (deterministic), with the real model downloads
 * (needs internet on the device/emulator; models are cached in the app's files dir between tests). Images:
 * ultralytics' bus.jpg / zidane.jpg (androidTest assets, AGPL like this project) and MediaPipe's sample hand photo,
 * downloaded at runtime (not committed), like the server's tests do.
 */
@RunWith(AndroidJUnit4::class)
class OnDevicePipelinesTest {
    companion object {
        private lateinit var context: Context
        private lateinit var loader: OnDeviceLoader
        private lateinit var captureRoot: File

        @BeforeClass
        @JvmStatic
        fun setUp() {
            context = InstrumentationRegistry.getInstrumentation().targetContext
            captureRoot = File(context.cacheDir, "captures-test")
            loader = OnDeviceLoader(context, ModelManager(File(context.filesDir, "models"), OkHttpDownloader()), CaptureStore(captureRoot))
        }

        @AfterClass
        @JvmStatic
        fun tearDown() {
            captureRoot.deleteRecursively()
        }

        const val HAND_PHOTOS = "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer"
    }

    private fun asset(name: String): Bitmap {
        val assets = InstrumentationRegistry.getInstrumentation().context.assets
        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        return assets.open(name).use { requireNotNull(BitmapFactory.decodeStream(it, null, options)) }
    }

    private fun downloaded(name: String): Bitmap {
        val file = File(context.cacheDir, "test-$name")
        if (!file.isFile) {
            val response = OkHttpClient().newCall(Request.Builder().url("$HAND_PHOTOS/$name").build()).execute()
            response.use {
                check(it.isSuccessful) { "download $name: HTTP ${it.code}" }
                file.writeBytes(it.body!!.bytes())
            }
        }
        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        return requireNotNull(BitmapFactory.decodeFile(file.path, options))
    }

    private fun defaults(id: String): Map<String, OptionValue> {
        val spec = LocalCatalog.specs.first { it.id == id }
        return spec.options.associate { it.key to it.default }
    }

    private fun run(id: String, image: Bitmap, options: Map<String, OptionValue> = defaults(id), gpu: Boolean = false): List<PixelDetection> {
        val pipeline = loader.load(id, options, gpu)
        try {
            return pipeline.run(BitmapFrame(image), 1, options)
        } finally {
            pipeline.close()
        }
    }

    @Test
    fun objects_find_person_and_bus_on_bus_jpg() {
        val labels = run(LocalCatalog.OBJECTS, asset("bus.jpg")).filter { it.conf >= 0.35f }.map { it.label }
        assertTrue(labels.toString(), "person" in labels)
        assertTrue(labels.toString(), "bus" in labels)
    }

    @Test
    fun objects_on_the_gpu_option_work_or_fall_back_to_the_cpu() {
        val labels = run(LocalCatalog.OBJECTS, asset("bus.jpg"), gpu = true).filter { it.conf >= 0.35f }.map { it.label }
        assertTrue(labels.toString(), "bus" in labels)
    }

    @Test
    fun pose_names_the_people_on_bus_jpg_standing() {
        val image = asset("bus.jpg")
        val people = run(LocalCatalog.POSE, image)
        // People with most of their body seen (the man cut off at the left edge doesn't count).
        val seen = people.filter { p -> p.points!!.count { it.v >= 0.5f } >= 12 }
        assertTrue("found ${people.map { it.label }}", seen.isNotEmpty())
        assertTrue(seen.map { it.label }.toString(), seen.any { it.label.startsWith("Standing") })
        val otherPostures = listOf("Sitting", "Squatting", "Kneeling", "Lying down", "Bending over")
        assertTrue(seen.map { it.label }.toString(), seen.none { p -> otherPostures.any { p.label.startsWith(it) } })
        assertTrue(people.all { it.detail == "person" && it.points!!.size == 17 })
        // COCO sides are the person's own: facing the camera, the left shoulder (5) is on the image's right.
        val facing = seen.filter { it.points!![5].v >= 0.5f && it.points!![6].v >= 0.5f }
        val leftOnRight = facing.count { it.points!![5].x > it.points!![6].x }
        assertTrue("left shoulder on the image right for $leftOnRight of ${facing.size}", leftOnRight * 2 > facing.size)

        val unnamed = run(LocalCatalog.POSE, image, defaults(LocalCatalog.POSE) + (LocalCatalog.OPT_NAMES to OptionValue.Bool(false)))
        assertTrue(unnamed.all { it.label == "person" })
    }

    @Test
    fun gestures_recognize_thumb_up() {
        val hands = run(LocalCatalog.GESTURES, downloaded("thumbs_up.jpg"))
        assertTrue(hands.map { it.label }.toString(), hands.any { it.label == "Thumb_Up" })
        assertTrue(hands.all { it.points!!.size == 21 && it.detail in setOf("Left", "Right") })
    }

    @Test
    fun faces_are_detected_on_zidane_jpg() {
        val faces = run(LocalCatalog.FACES, asset("zidane.jpg"))
        assertTrue(faces.size >= 1)
        assertTrue(faces.all { it.label == "face" && it.x2 > it.x1 && it.y2 > it.y1 })
    }

    @Test
    fun text_reads_a_sign() {
        val sign = Bitmap.createBitmap(640, 360, Bitmap.Config.ARGB_8888)
        Canvas(sign).apply {
            drawColor(Color.WHITE)
            val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.BLACK
                textSize = 110f
                typeface = Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)
            }
            drawText("EXIT 42", 90f, 220f, paint)
        }
        val lines = run(LocalCatalog.TEXT, sign)
        assertTrue(lines.map { it.label }.toString(), lines.any { "EXIT" in it.label.uppercase() })
    }

    @Test
    fun pose_capture_writes_the_server_files() {
        val options = defaults(LocalCatalog.POSE)
        val pose = loader.load(LocalCatalog.POSE, options, false)
        try {
            pose.prepare(LocalCatalog.ACTION_CAPTURE) // downloads hand_landmarker.task on first use
            val image = asset("bus.jpg")
            val outcome = pose.command(LocalCatalog.ACTION_CAPTURE, emptyMap(), BitmapFrame(image), 1, options)
            assertTrue(outcome.message, outcome.ok)
            assertEquals(CaptureStore.FILES, outcome.files.map { it.name to it.kind })
            val files = outcome.files.map { requireNotNull(CaptureFiles.localSource(it.url, captureRoot)) { it.url } }
            assertTrue(files.all { it.length() > 0 })
            val json = JSONObject(files.first { it.name == "pose_keypoints.json" }.readText())
            assertEquals(image.height, json.getInt("canvas_height"))
            assertEquals(image.width, json.getInt("canvas_width"))
            val people = json.getJSONArray("people")
            assertTrue(outcome.message, outcome.message.startsWith("Captured ${people.length()} "))
            assertEquals(54, people.getJSONObject(0).getJSONArray("pose_keypoints_2d").length())
            val png = BitmapFactory.decodeFile(files.first { it.name == "openpose_full.png" }.path)
            assertEquals(image.width to image.height, png.width to png.height)
        } finally {
            pose.close()
        }
    }

    /** The whole engine as the app runs it: real lanes, loading on enable, a frame in, a server-shaped reply out. */
    @Test
    fun local_backend_end_to_end() {
        val ready = CountDownLatch(1)
        val answered = CountDownLatch(1)
        var reply: ServerMessage.Detections? = null
        val listener = object : RecognitionBackend.Listener {
            override fun onStateChanged(state: ConnectionState, detail: String?) = Unit
            override fun onHello(hello: ServerMessage.Hello) = Unit
            override fun onPipelineStatus(change: ServerMessage.PipelineStatusChange) {
                if (change.pipeline == LocalCatalog.OBJECTS && change.status == PipelineStatus.READY) ready.countDown()
            }
            override fun onDetections(result: ServerMessage.Detections, receivedAtMs: Long) {
                reply = result
                answered.countDown()
            }
            override fun onCommandResult(result: ServerMessage.CommandResult) = Unit
        }
        val pools = List(3) { Executors.newSingleThreadExecutor() }
        val backend = LocalBackend(
            listener, StreamStats(), loader, LocalLanes(pools[0], pools[1], pools[2]) { pools.forEach { it.shutdown() } }, "test",
        )
        try {
            backend.start()
            assertTrue("objects never became ready", ready.await(120, TimeUnit.SECONDS))
            assertTrue(backend.offer(BitmapFrame(asset("bus.jpg"))))
            assertTrue(answered.await(30, TimeUnit.SECONDS))
            assertNotNull(reply)
            val r = reply!!
            assertTrue(r.pipelineMs.containsKey(LocalCatalog.OBJECTS))
            assertTrue(r.tracks.any { it.label == "bus" && it.trackId!!.startsWith("objects:") })
        } finally {
            backend.shutdown()
        }
    }
}
