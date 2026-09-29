package ir.platereader.core

import ai.onnxruntime.OrtEnvironment
import org.json.JSONObject
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File
import java.util.Base64
import javax.imageio.ImageIO
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Checks that the Kotlin pipeline (what the Android app runs) gives the same results as the Python reference
 * (`platereader.pipeline`, ONNX backend) on real test data. Fixtures: tools/make_parity_fixtures.py.
 * Paths come from system properties set in core/build.gradle.kts (repo models/ and data/).
 */
class ParityTest {
    companion object {
        private lateinit var engine: PlateEngine
        private lateinit var fx: JSONObject
        private val dataDir = File(System.getProperty("plate.data.dir", "../../data"))

        @JvmStatic
        @BeforeClass
        fun setUp() {
            val models = File(System.getProperty("plate.models.dir", "../../models"))
            fx = JSONObject(ParityTest::class.java.getResource("/parity.json")!!.readText())
            val c = fx.getJSONObject("config")
            val letters = c.getJSONArray("letter_vocab").let { a -> List(a.length()) { a.getString(it) } }
            val latin = c.getJSONArray("letter_vocab_latin").let { a -> List(a.length()) { a.getString(it) } }
            val cfg = PlateConfig(c.getInt("yolo_imgsz"), c.getDouble("yolo_conf_threshold").toFloat(),
                                  c.getDouble("crop_padding").toFloat(), letters, latin)
            engine = PlateEngine(OrtEnvironment.getEnvironment(), File(models, fx.getString("det_model")).readBytes(),
                                 File(models, fx.getString("rec_model")).readBytes(), cfg, threads = 4)
        }

        @JvmStatic
        @AfterClass
        fun tearDown() = engine.close()

        fun load(f: File): RgbImage {
            val bi = ImageIO.read(f)
            return RgbImage(bi.width, bi.height, bi.getRGB(0, 0, bi.width, bi.height, null, 0, bi.width))
        }

        fun fromRgbBytes(b: ByteArray, w: Int, h: Int) = RgbImage(w, h, IntArray(w * h) {
            (0xFF shl 24) or ((b[3 * it].toInt() and 0xFF) shl 16) or ((b[3 * it + 1].toInt() and 0xFF) shl 8) or (b[3 * it + 2].toInt() and 0xFF)
        })
    }

    @Test
    fun letterboxIsPillowExact() {
        val refs = fx.getJSONArray("letterbox_pixels")
        for (i in 0 until refs.length()) {
            val r = refs.getJSONObject(i)
            val src = fromRgbBytes(Base64.getDecoder().decode(r.getString("src_base64")), r.getInt("src_w"), r.getInt("src_h"))
            val expected = Base64.getDecoder().decode(r.getString("rgb_base64"))
            val got = engine.letterboxPlate(src)
            var maxDiff = 0
            for (p in 0 until 64 * 256) {
                maxDiff = max(maxDiff, abs(got.r(p) - (expected[3 * p].toInt() and 0xFF)))
                maxDiff = max(maxDiff, abs(got.g(p) - (expected[3 * p + 1].toInt() and 0xFF)))
                maxDiff = max(maxDiff, abs(got.b(p) - (expected[3 * p + 2].toInt() and 0xFF)))
            }
            println("letterbox ${r.getString("file")} (${src.width}x${src.height}): max |kotlin - pillow| = $maxDiff")
            assertEquals("Pillow-exact letterbox", 0, maxDiff)
        }
    }

    @Test
    fun recognizerMatchesPython() {
        val crops = fx.getJSONArray("crops")
        assumeTrue("LPR data not found in $dataDir", File(dataDir, "LPR/detections").isDirectory)
        var same = 0; var correct = 0
        for (i in 0 until crops.length()) {
            val c = crops.getJSONObject(i)
            val (text, _, _) = engine.recognize(listOf(load(File(dataDir, "LPR/detections/" + c.getString("file")))))[0]
            if (text == c.getString("text")) same++
            if (text == c.getString("label")) correct++
        }
        val n = crops.length()
        println("recognizer on $n test crops: same text as Python ${same}/$n, correct ${correct}/$n")
        assertTrue("agreement with Python ${same}/$n", same >= 0.97 * n)
    }

    @Test
    fun endToEndMatchesPython() {
        val imgs = fx.getJSONArray("images")
        assumeTrue("LPD data not found in $dataDir", File(dataDir, "LPD/images").isDirectory)
        var pyPlates = 0; var matched = 0; var sameText = 0; var ious = 0.0; var ms = 0L
        for (i in 0 until imgs.length()) {
            val e = imgs.getJSONObject(i)
            val img = load(File(dataDir, "LPD/images/" + e.getString("file")))
            val t0 = System.currentTimeMillis()
            val got = engine.read(img)
            ms += System.currentTimeMillis() - t0
            val ref = e.getJSONArray("plates")
            for (j in 0 until ref.length()) {
                pyPlates++
                val r = ref.getJSONObject(j)
                val b = r.getJSONArray("box").let { a -> FloatArray(4) { a.getDouble(it).toFloat() } }
                val best = got.maxByOrNull { iou(it.box, b) } ?: continue
                val v = iou(best.box, b)
                if (v > 0.5) {
                    matched++; ious += v
                    if (best.text == r.getString("text")) sameText++
                }
            }
        }
        println("end-to-end on ${imgs.length()} images: plates matched $matched/$pyPlates, same text $sameText/$matched, " +
                "mean IoU vs Python ${"%.4f".format(ious / max(matched, 1))}, ${ms / imgs.length()} ms/image (JVM)")
        assertTrue("detections matched", matched >= 0.95 * pyPlates)
        assertTrue("texts agree", sameText >= 0.95 * matched)
        assertTrue("boxes agree", ious / max(matched, 1) > 0.95)
    }

    @Test
    fun persianDisplay() {
        assertEquals("۱۲ ب ۳۴۵ | ۶۷", PlateText.display("12ب34567"))
        assertEquals("۶۴ الف ۷۳۹ | ۱۱", PlateText.display("64الف73911"))
    }

    private fun iou(a: Detection, b: FloatArray): Double {
        val iw = max(0f, min(a.x2, b[2]) - max(a.x1, b[0])); val ih = max(0f, min(a.y2, b[3]) - max(a.y1, b[1]))
        val inter = (iw * ih).toDouble()
        return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    }
}
