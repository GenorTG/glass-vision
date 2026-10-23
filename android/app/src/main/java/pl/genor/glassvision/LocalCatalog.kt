package pl.genor.glassvision

/**
 * A model file the on-device engine downloads on first use, from Google's original URL (the versioned `/1/` path,
 * so the recorded size + sha256 stay valid; `latest` has already moved for pose_landmarker_full). Sizes/hashes
 * were measured on 2026-10-23 from these URLs.
 */
enum class LocalModel(
    val fileName: String,
    val title: String,
    val url: String,
    val bytes: Long,
    val sha256: String,
) {
    // https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector#models (COCO, 80 classes)
    EFFICIENTDET_LITE0(
        "efficientdet_lite0.tflite", "Objects · EfficientDet-Lite0",
        "$BUCKET/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite",
        7_254_339, "4b59100025bea1235a84c1038879a6cccc9f6c49f5e41144e91e74d99e780993",
    ),
    EFFICIENTDET_LITE2(
        "efficientdet_lite2.tflite", "Objects · EfficientDet-Lite2",
        "$BUCKET/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.tflite",
        12_138_859, "5d4ebec1029bc9907aeadb9e7b4ac9cb1da6a19d01ad375210a9ae18ba173302",
    ),
    // https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer#models
    GESTURE_RECOGNIZER(
        "gesture_recognizer.task", "Hand gestures",
        "$BUCKET/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        8_373_440, "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482",
    ),
    // https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models
    POSE_LITE(
        "pose_landmarker_lite.task", "Body pose · lite",
        "$BUCKET/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        5_777_746, "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    ),
    POSE_FULL(
        "pose_landmarker_full.task", "Body pose · full",
        "$BUCKET/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        9_398_198, "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1",
    ),
    // https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector#models
    BLAZE_FACE(
        "blaze_face_short_range.tflite", "Faces · BlazeFace",
        "$BUCKET/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        229_746, "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
    ),
    // https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models (pose capture: hands)
    HAND_LANDMARKER(
        "hand_landmarker.task", "Hands for pose capture",
        "$BUCKET/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        7_819_105, "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
}

private const val BUCKET = "https://storage.googleapis.com/mediapipe-models"

/**
 * What the on-device engine offers, described exactly like a server's `hello` (docs/PROTOCOL.md), so the app's
 * chips, cards, option editors and actions work unchanged. Ids match the server where the feature matches;
 * `open_vocab` is missing on purpose (it needs a 570 MB text encoder). Pure.
 */
object LocalCatalog {
    const val OBJECTS = "objects"
    const val GESTURES = "gestures"
    const val POSE = "pose"
    const val FACES = "faces"
    const val TEXT = "text"

    const val OPT_MODEL = "model"
    const val OPT_NAMES = "names"
    const val OPT_NUM_HANDS = "num_hands"
    const val OPT_MIN_CONF = "min_conf"
    const val OPT_TRANSLATE_TO = "translate_to"
    const val OPT_OCR_LANGUAGE = "ocr_language"
    const val ACTION_CAPTURE = "capture"

    const val OBJECTS_FAST = "Lite0 · fast"
    const val OBJECTS_ACCURATE = "Lite2 · higher accuracy"
    const val POSE_LITE = "Lite · fast"
    const val POSE_FULL = "Full · higher accuracy"
    const val TRANSLATE_OFF = "Off"
    val ocrLanguageChoices = OcrLanguage.entries.map { it.choice }

    /** ML Kit translation language tags; language models download on demand and work offline afterwards. */
    val translationChoices = listOf(
        TRANSLATE_OFF,
        "Arabic · ar", "Bengali · bn", "Bulgarian · bg", "Catalan · ca", "Chinese · zh", "Croatian · hr",
        "Czech · cs", "Danish · da", "Dutch · nl", "English · en", "Estonian · et", "Finnish · fi",
        "French · fr", "German · de", "Greek · el", "Gujarati · gu", "Hebrew · he", "Hindi · hi",
        "Hungarian · hu", "Indonesian · id", "Italian · it", "Japanese · ja", "Kannada · kn", "Korean · ko",
        "Latvian · lv", "Lithuanian · lt", "Malay · ms", "Marathi · mr", "Norwegian · no", "Persian · fa",
        "Polish · pl", "Portuguese · pt", "Romanian · ro", "Russian · ru", "Slovak · sk", "Slovenian · sl",
        "Spanish · es", "Swahili · sw", "Swedish · sv", "Tamil · ta", "Telugu · te", "Thai · th",
        "Turkish · tr", "Ukrainian · uk", "Urdu · ur", "Vietnamese · vi",
    )

    fun translationTag(choice: String): String? = choice.substringAfter(" · ", "").takeIf { it.isNotBlank() }

    /** OCR runs in the background on the newest frame this often (ML Kit takes ~100–300 ms on a phone). */
    const val TEXT_INTERVAL_MS = 500

    /** MediaPipe HAND_CONNECTIONS, same as the server's gestures skeleton. */
    val HAND_SKELETON = listOf(
        0 to 1, 0 to 5, 9 to 13, 13 to 17, 5 to 9, 0 to 17,
        1 to 2, 2 to 3, 3 to 4,
        5 to 6, 6 to 7, 7 to 8,
        9 to 10, 10 to 11, 11 to 12,
        13 to 14, 14 to 15, 15 to 16,
        17 to 18, 18 to 19, 19 to 20,
    )

    /** COCO-17 skeleton, same as the server's pose pipeline. */
    val COCO_SKELETON = listOf(
        15 to 13, 13 to 11, 16 to 14, 14 to 12, 11 to 12, 5 to 11, 6 to 12, 5 to 6, 5 to 7, 6 to 8,
        7 to 9, 8 to 10, 1 to 2, 0 to 1, 0 to 2, 1 to 3, 2 to 4, 3 to 5, 4 to 6,
    )

    val specs: List<PipelineSpec> = listOf(
        PipelineSpec(
            id = OBJECTS,
            name = "Objects (80 everyday classes)",
            description = "People, vehicles, animals, furniture, food, electronics… (EfficientDet-Lite, COCO).",
            defaultEnabled = true,
            usesGlobalConf = true,
            options = listOf(
                OptionSpec(OPT_MODEL, "Model", OptionType.CHOICE, OptionValue.Str(OBJECTS_FAST), choices = listOf(OBJECTS_FAST, OBJECTS_ACCURATE)),
            ),
        ),
        PipelineSpec(
            id = GESTURES,
            name = "Hand gestures",
            description = "Hands (21 landmarks) and 7 gestures: thumb up, thumb down, open palm, fist, pointing up, " +
                "victory, I-love-you. Unrecognized poses show as 'hand'.",
            options = listOf(
                OptionSpec(OPT_NUM_HANDS, "Max hands", OptionType.NUMBER, OptionValue.Num(2.0), min = 1.0, max = 4.0, step = 1.0),
                OptionSpec(OPT_MIN_CONF, "Min hand confidence", OptionType.NUMBER, OptionValue.Num(0.5), min = 0.1, max = 0.9, step = 0.05),
            ),
            skeleton = HAND_SKELETON,
        ),
        PipelineSpec(
            id = POSE,
            name = "Body pose",
            description = "Skeleton of every person, named by posture and arms (e.g. 'Standing · Arms up'; MediaPipe Pose). " +
                "Capture saves OpenPose images + JSON (body and hands) for ComfyUI ControlNet.",
            options = listOf(
                OptionSpec(OPT_NAMES, "Pose names", OptionType.BOOL, OptionValue.Bool(true)),
                OptionSpec(OPT_MODEL, "Model", OptionType.CHOICE, OptionValue.Str(POSE_LITE), choices = listOf(POSE_LITE, POSE_FULL)),
            ),
            actions = listOf(ActionSpec(ACTION_CAPTURE, "Capture pose")),
            skeleton = COCO_SKELETON,
        ),
        PipelineSpec(
            id = FACES,
            name = "Faces",
            description = "Detects faces (BlazeFace). Naming people needs the server.",
        ),
        PipelineSpec(
            id = TEXT,
            name = "Text (OCR)",
            description = "Reads 41 languages offline with downloadable language packs; optionally translates text offline.",
            options = listOf(
                OptionSpec(
                    OPT_OCR_LANGUAGE,
                    "OCR language",
                    OptionType.CHOICE,
                    OptionValue.Str(OcrLanguage.ENGLISH.choice),
                    choices = ocrLanguageChoices,
                ),
                OptionSpec(
                    OPT_TRANSLATE_TO,
                    "Translate to",
                    OptionType.CHOICE,
                    OptionValue.Str(TRANSLATE_OFF),
                    choices = translationChoices,
                ),
            ),
            intervalMs = TEXT_INTERVAL_MS,
        ),
    )

    /** The model a pipeline needs with [options] (resolved, see PipelineSettings.resolve); null = bundled/none. */
    fun modelFor(pipeline: String, options: Map<String, OptionValue>): LocalModel? = when (pipeline) {
        OBJECTS -> if ((options[OPT_MODEL] as? OptionValue.Str)?.value == OBJECTS_ACCURATE) {
            LocalModel.EFFICIENTDET_LITE2
        } else {
            LocalModel.EFFICIENTDET_LITE0
        }
        GESTURES -> LocalModel.GESTURE_RECOGNIZER
        POSE -> if ((options[OPT_MODEL] as? OptionValue.Str)?.value == POSE_FULL) LocalModel.POSE_FULL else LocalModel.POSE_LITE
        FACES -> LocalModel.BLAZE_FACE
        else -> null
    }

    /** `hello` for the on-device engine; [device] like a server's ("This phone (GPU)"), statuses per pipeline. */
    fun hello(version: String, device: String, statuses: Map<String, Pair<PipelineStatus, String?>>): ServerMessage.Hello =
        ServerMessage.Hello(
            protocol = Protocol.VERSION,
            device = device,
            version = version,
            pipelines = specs.map { spec ->
                val (status, message) = statuses[spec.id] ?: (PipelineStatus.NOT_LOADED to null)
                spec.copy(status = status, statusMessage = message)
            },
        )
}

/**
 * BlazePose's 33 landmarks → the 17 COCO keypoints the rest of the app (skeleton, pose names, OpenPose export)
 * uses. MediaPipe names landmarks by the person's own sides like COCO does.
 * https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model
 */
object BlazePose {
    const val LANDMARKS = 33

    /** COCO index → BlazePose index: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles. */
    val COCO_FROM_BLAZE = intArrayOf(0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

    /** [landmarks] normalized (x, y, visibility) → COCO-17 in pixels of a [width]×[height] frame. */
    fun toCoco(landmarks: List<KeyPoint>, width: Int, height: Int): List<KeyPoint> {
        require(landmarks.size >= LANDMARKS) { "BlazePose has $LANDMARKS landmarks, got ${landmarks.size}" }
        return COCO_FROM_BLAZE.map { i ->
            val p = landmarks[i]
            KeyPoint(p.x * width, p.y * height, p.v)
        }
    }

    /**
     * Person box in pixels from all landmarks seen with visibility ≥ [minVisibility] (feet and hands included),
     * padded by 10% per side like the server's hand boxes, clipped to the frame; null if fewer than 2 seen.
     */
    fun box(landmarks: List<KeyPoint>, width: Int, height: Int, minVisibility: Float = 0.3f): FloatArray? {
        val seen = landmarks.filter { it.v >= minVisibility }
        if (seen.size < 2) return null
        val xs = seen.map { it.x * width }
        val ys = seen.map { it.y * height }
        val padX = (xs.max() - xs.min()) * BOX_PAD
        val padY = (ys.max() - ys.min()) * BOX_PAD
        return floatArrayOf(
            (xs.min() - padX).coerceIn(0f, width.toFloat()),
            (ys.min() - padY).coerceIn(0f, height.toFloat()),
            (xs.max() + padX).coerceIn(0f, width.toFloat()),
            (ys.max() + padY).coerceIn(0f, height.toFloat()),
        )
    }

    const val BOX_PAD = 0.10f
}
