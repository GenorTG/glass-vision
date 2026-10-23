package pl.genor.glassvision

/** A downloadable Tesseract LSTM language pack. Metadata is pinned to tessdata_fast commit 8741641. */
enum class OcrLanguage(
    val code: String,
    val title: String,
    val bytes: Long,
    val sha256: String,
) {
    ENGLISH("eng", "English", 4_113_088, "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2"),
    POLISH("pol", "Polish", 4_765_518, "c4476cdbc0e33d898d32345122b7be1cbf85ace15f920f06c7714756e1ef79b2"),
    GERMAN("deu", "German", 1_525_436, "19d219bbb6672c869d20a9636c6816a81eb9a71796cb93ebe0cb1530e2cdb22d"),
    FRENCH("fra", "French", 1_130_365, "ced037562e8c80c13122dece28dd477d399af80911a28791a66a63ac1e3445ca"),
    SPANISH("spa", "Spanish", 2_294_433, "6f2e04d02774a18f01bed44b1111f2cd7f3ba7ac9dc4373cd3f898a40ea6b464"),
    ITALIAN("ita", "Italian", 2_701_314, "b8f89e1e785118dac4d51ae042c029a64edb5c3ee42ef73027a6d412748d8827"),
    PORTUGUESE("por", "Portuguese", 1_982_756, "c4932b937207a9514b7514d518b931a99938c02a28a5a5a553f8599ed58b7deb"),
    ARABIC("ara", "Arabic", 1_432_056, "e3206d3dc87fd50c24a0fb9f01838615911d25168f4e64415244b67d2bb3e729"),
    HINDI("hin", "Hindi", 1_122_751, "4c73ffc59d497c186b19d1e90f5d721d678ea6b2e277b719bee4e2af12271825"),
    CHINESE_SIMPLIFIED("chi_sim", "Chinese (simplified)", 2_469_156, "a5fcb6f0db1e1d6d8522f39db4e848f05984669172e584e8d76b6b3141e1f730"),
    CHINESE_TRADITIONAL("chi_tra", "Chinese (traditional)", 2_366_642, "529c5b5797d64b126065cd55f2bb4c7fd7b15790798091b1ff259941a829330b"),
    JAPANESE("jpn", "Japanese", 2_471_260, "1f5de9236d2e85f5fdf4b3c500f2d4926f8d9449f28f5394472d9e8d83b91b4d"),
    KOREAN("kor", "Korean", 1_677_415, "6b85e11d9bbf07863b97b3523b1b112844c43e713df8b66418a081fd1060b3b2"),
    RUSSIAN("rus", "Russian", 3_861_738, "e16e5e036cce1d9ec2b00063cf8b54472625b9e14d893a169e2b0dedeb4df225"),
    UKRAINIAN("ukr", "Ukrainian", 3_825_102, "d59e53e2bded32f4445f124b4b00240fcac7e8044c003ab822ccb94f0b3db59b"),
    GREEK("ell", "Greek", 1_419_514, "4fba8a0b461038d51f1c20d043d4f2ac38c4e778f1b90830847f7bd8fa3ba726"),
    HEBREW("heb", "Hebrew", 961_404, "11f9e43ab227f786352a50f75c94c2e9906f1baba86d93276da19da7ce0904db"),
    THAI("tha", "Thai", 1_072_600, "294227cc2d1292b0acb28d61d4115c88252b96d466ca90b417cf4cf0c67bf07c"),
    BENGALI("ben", "Bengali", 855_841, "31163084c279aaebd376216f0c3d5c17ad4b5fee8db49dae79c20000b5de5964"),
    TAMIL("tam", "Tamil", 3_237_963, "d02fbec24be4b07e32e80d0ccfc3b6b67a3c5d61c9d0a7c8532677990912c6ec"),
    VIETNAMESE("vie", "Vietnamese", 531_275, "79df64caf7bcfb2a27df5042ecb6121e196eada34da774956995747636d5bfa1"),
    TURKISH("tur", "Turkish", 4_550_554, "7393381111e1152420fc4092cb44eef4237580d21b92bf30d7d221aad192c6b7"),
    INDONESIAN("ind", "Indonesian", 1_122_661, "69786901da87ab8766c1ea7fbb10b28f2110c14da3f6c8f2735df131fba95d88"),
    SWEDISH("swe", "Swedish", 4_167_034, "f7304988d41f833efebcc2d529df54b1903ecebbc3da1faabd19a0fddd4fe586"),
    DANISH("dan", "Danish", 2_580_059, "acb1fd074487a31d1294fcdfd7d7c673467ffd8aeacb2ccd61ebcbf04eb4e2fa"),
    DUTCH("nld", "Dutch", 6_050_296, "ced0e5e046a84c908a6aa7accbef9a232c4a5d9a8276691b81c6ee64d02963f6"),
    CZECH("ces", "Czech", 3_795_684, "934bcaf97ef3348413263331131c9fa7f55f30db333c711929c124fb635f7e1b"),
    ROMANIAN("ron", "Romanian", 2_376_323, "9adfde6b51ba4b97efd10ea37c3070fd3fc2bad7815e81f5c3c198cd96216cc9"),
    HUNGARIAN("hun", "Hungarian", 5_296_273, "35067e7cfe102dcdc953f9a758fdfaa6296b17a1ee6d874ee780fa306430b9fb"),
    BULGARIAN("bul", "Bulgarian", 1_675_212, "aebc9b0fcc8cfaf8a9f38a02bb7b85052bd850744696a2c11cf0081820e5b21e"),
    SERBIAN("srp", "Serbian", 2_149_931, "aa41ae3d9cc705e60d398ab38a5c3cc8b772c0d420c7d4f0859beb13d0e321b6"),
    BOSNIAN("bos", "Bosnian", 2_498_242, "6cc8cc87cf1afbfa6a41febb725dbadb14bed96a46685906440a6eb8a7892f04"),
    CROATIAN("hrv", "Croatian", 4_103_348, "9e515d9832ce259dbab550b1cc6b998f8b929faf2edacaaca981b05adb130571"),
    SLOVAK("slk", "Slovak", 4_427_661, "fbcc400a9c74c6a13d922fcb1211b655d1b165387b675ed75cd2dbd756b974a5"),
    SLOVENIAN("slv", "Slovenian", 3_003_829, "b937632c17ce5fdf20535d25feec044da10a0c4c8234f4302d5606fd671cf60d"),
    FINNISH("fin", "Finnish", 7_865_732, "61a04cd62b507c3d9ae0e1cda399e6715ebf49dea9df47897c8acdcd3bd3e13c"),
    NORWEGIAN("nor", "Norwegian", 3_610_079, "0451eb4f8049ae78196806bf878a389a2f40f1386fe038568cf4441226ba6ef2"),
    ESTONIAN("est", "Estonian", 4_458_101, "515d4a773682b286369511e83fe412bcff16a92a886f99c761c1d760a7e30456"),
    LATVIAN("lav", "Latvian", 2_717_247, "7d9eaf22254b381f18b806cd6cb647862a397a7221adbce3e22cc421793a8847"),
    LITHUANIAN("lit", "Lithuanian", 3_154_896, "1e383df5b055583bc01cb5764ecdf74c540753f2cb3f8205e7105361da4bc989"),
    MALAY("msa", "Malay", 1_747_801, "e41a3e5febfec50c90371eb1cbb17a48b10cad387900e3420b1f134c1b766cba"),
    ;

    companion object {
        const val AUTO_CHOICE = "Auto-detect"
        private const val BASE_URL = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/87416418657359cb625c412a48b6e1d6d41c29bd"

        fun fromChoice(choice: String): OcrLanguage = entries.firstOrNull { it.choice == choice } ?: ENGLISH

        /** ML Kit language tags for languages with a matching on-device Tesseract pack. */
        fun fromLanguageTag(tag: String): OcrLanguage? = when (tag) {
            "en" -> ENGLISH
            "pl" -> POLISH
            "de" -> GERMAN
            "fr" -> FRENCH
            "es" -> SPANISH
            "it" -> ITALIAN
            "pt" -> PORTUGUESE
            "ar" -> ARABIC
            "hi" -> HINDI
            "zh" -> CHINESE_SIMPLIFIED
            "ja" -> JAPANESE
            "ko" -> KOREAN
            "ru" -> RUSSIAN
            "uk" -> UKRAINIAN
            "el" -> GREEK
            "he" -> HEBREW
            "th" -> THAI
            "bn" -> BENGALI
            "ta" -> TAMIL
            "vi" -> VIETNAMESE
            "tr" -> TURKISH
            "id" -> INDONESIAN
            "sv" -> SWEDISH
            "da" -> DANISH
            "nl" -> DUTCH
            "cs" -> CZECH
            "ro" -> ROMANIAN
            "hu" -> HUNGARIAN
            "bg" -> BULGARIAN
            "sr" -> SERBIAN
            "hr" -> CROATIAN
            "sk" -> SLOVAK
            "sl" -> SLOVENIAN
            "fi" -> FINNISH
            "no" -> NORWEGIAN
            "et" -> ESTONIAN
            "lv" -> LATVIAN
            "lt" -> LITHUANIAN
            "ms" -> MALAY
            else -> null
        }
    }

    val choice: String get() = "$title · $code"
    val spec: ModelSpec get() = ModelSpec(
        id = "ocr:$code",
        fileName = "tessdata/$code.traineddata",
        title = "OCR · $title",
        url = "$BASE_URL/$code.traineddata",
        bytes = bytes,
        sha256 = sha256,
    )

}
