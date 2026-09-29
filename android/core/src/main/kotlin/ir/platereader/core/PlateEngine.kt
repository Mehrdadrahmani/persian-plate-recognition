package ir.platereader.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/** Settings read from the model bundle (generated from models/lpr_config.json at build time). */
data class PlateConfig(
    val imgsz: Int,
    val conf: Float,
    val cropPadding: Float,
    val letters: List<String>,
    val latin: List<String>,
    val iou: Float = 0.7f,
    val recH: Int = 64,
    val recW: Int = 256,
    val padValue: Int = 114,
    val mean: FloatArray = floatArrayOf(0.485f, 0.456f, 0.406f),
    val std: FloatArray = floatArrayOf(0.229f, 0.224f, 0.225f),
) {
    companion object {
        /** `pipeline.properties` (key=value lines) + `letters.tsv` (char<TAB>latin per line). */
        fun parse(properties: String, lettersTsv: String): PlateConfig {
            val p = properties.lines().filter { '=' in it && !it.trimStart().startsWith("#") }
                .associate { it.substringBefore('=').trim() to it.substringAfter('=').trim() }
            val rows = lettersTsv.lines().filter { it.isNotBlank() }.map { it.split('\t') }
            return PlateConfig(
                imgsz = p.getValue("yolo_imgsz").toInt(),
                conf = p.getValue("yolo_conf_threshold").toFloat(),
                cropPadding = p.getValue("crop_padding").toFloat(),
                letters = rows.map { it[0] },
                latin = rows.map { it.getOrElse(1) { "?" } },
            )
        }
    }
}

data class Detection(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float)

data class PlateResult(
    val box: Detection,
    val text: String,
    val textConf: Float,
    val charConf: FloatArray,
) {
    val persian: String get() = PlateText.display(text)
    fun latin(cfg: PlateConfig) = PlateText.latin(text, cfg.letters, cfg.latin)
}

/**
 * End-to-end plate reader on ONNX Runtime: YOLO26 detection -> padded crop -> CNN recognition.
 * A line-by-line port of the Python reference `platereader.pipeline` (backend="onnx").
 */
class PlateEngine(
    private val env: OrtEnvironment,
    detectorModel: ByteArray,
    recognizerModel: ByteArray,
    val config: PlateConfig,
    threads: Int = 4,
    /** extra session setup, e.g. `{ it.addXnnpack(mapOf("intra_op_num_threads" to "4")) }` on Android */
    configure: (OrtSession.SessionOptions) -> Unit = {},
) : AutoCloseable {
    private val det: OrtSession
    private val rec: OrtSession
    var lastDetectMs = 0L; private set
    var lastRecognizeMs = 0L; private set

    init {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            configure(this)
        }
        det = env.createSession(detectorModel, opts)
        rec = env.createSession(recognizerModel, opts)
    }

    // ---------------------------------------------------------------- detection
    private class Letterboxed(val tensor: FloatArray, val ratio: Double, val padX: Int, val padY: Int)

    /** Ultralytics LetterBox(auto=False) with OpenCV INTER_LINEAR resize, RGB 0-1, NCHW. */
    private fun letterboxYolo(img: RgbImage): Letterboxed {
        val size = config.imgsz
        val r = min(size.toDouble() / img.height, size.toDouble() / img.width)
        val nw = Math.rint(img.width * r).toInt()
        val nh = Math.rint(img.height * r).toInt()
        val dw = (size - nw) / 2.0
        val dh = (size - nh) / 2.0
        val resized = if (nw != img.width || nh != img.height) ImageOps.resizeCvLinear(img, nw, nh) else img
        val top = Math.rint(dh - 0.1).toInt()
        val left = Math.rint(dw - 0.1).toInt()
        val canvas = RgbImage.filled(size, size, config.padValue)
        ImageOps.paste(canvas, resized, left, top)
        val plane = size * size
        val t = FloatArray(3 * plane)
        for (i in 0 until plane) {
            val p = canvas.argb[i]
            t[i] = ((p shr 16) and 0xFF) / 255f
            t[plane + i] = ((p shr 8) and 0xFF) / 255f
            t[2 * plane + i] = (p and 0xFF) / 255f
        }
        return Letterboxed(t, r, left, top)
    }

    fun detect(img: RgbImage, conf: Float = config.conf): List<Detection> {
        val t0 = System.currentTimeMillis()
        val lb = letterboxYolo(img)
        val size = config.imgsz.toLong()
        val out: FloatArray
        val n: Int
        OnnxTensor.createTensor(env, FloatBuffer.wrap(lb.tensor), longArrayOf(1, 3, size, size)).use { input ->
            det.run(mapOf(det.inputNames.first() to input)).use { res ->
                val o = res[0] as OnnxTensor
                n = o.info.shape[2].toInt()  // [1, 5, N]: cx, cy, w, h, score
                out = FloatArray(5 * n).also { o.floatBuffer.get(it) }
            }
        }
        val cand = ArrayList<Detection>()
        for (i in 0 until n) {
            val s = out[4 * n + i]
            if (s < conf) continue
            val cx = out[i]; val cy = out[n + i]; val w = out[2 * n + i]; val h = out[3 * n + i]
            cand += Detection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, s)
        }
        val kept = nms(cand, config.iou)
        val W = img.width.toFloat(); val H = img.height.toFloat()
        lastDetectMs = System.currentTimeMillis() - t0
        return kept.map {
            Detection(((it.x1 - lb.padX) / lb.ratio).toFloat().coerceIn(0f, W), ((it.y1 - lb.padY) / lb.ratio).toFloat().coerceIn(0f, H),
                      ((it.x2 - lb.padX) / lb.ratio).toFloat().coerceIn(0f, W), ((it.y2 - lb.padY) / lb.ratio).toFloat().coerceIn(0f, H), it.score)
        }
    }

    private fun nms(boxes: List<Detection>, iouThr: Float, maxDet: Int = 300): List<Detection> {
        val order = boxes.sortedByDescending { it.score }.toMutableList()
        val keep = ArrayList<Detection>()
        while (order.isNotEmpty() && keep.size < maxDet) {
            val a = order.removeAt(0)
            keep += a
            order.removeAll { b ->
                val iw = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
                val ih = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
                val inter = iw * ih
                val union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
                inter / (union + 1e-9f) > iouThr
            }
        }
        return keep
    }

    // ---------------------------------------------------------------- recognition
    /** Box enlarged by cropPadding x size on each side, rounded like Python's round(), clamped to the image. */
    fun cropWithPadding(img: RgbImage, d: Detection): RgbImage? {
        val bw = d.x2 - d.x1; val bh = d.y2 - d.y1; val pad = config.cropPadding
        val x1 = max(0, Math.rint((d.x1 - pad * bw).toDouble()).toInt())
        val y1 = max(0, Math.rint((d.y1 - pad * bh).toDouble()).toInt())
        val x2 = min(img.width, Math.rint((d.x2 + pad * bw).toDouble()).toInt())
        val y2 = min(img.height, Math.rint((d.y2 + pad * bh).toDouble()).toInt())
        return if (x2 - x1 >= 2 && y2 - y1 >= 2) ImageOps.crop(img, x1, y1, x2, y2) else null
    }

    /** lpr_models.letterbox: keep aspect, Pillow BILINEAR, centre on grey. */
    fun letterboxPlate(crop: RgbImage): RgbImage {
        val H = config.recH; val W = config.recW
        val s = min(W.toDouble() / crop.width, H.toDouble() / crop.height)
        val nw = max(1, Math.rint(crop.width * s).toInt())
        val nh = max(1, Math.rint(crop.height * s).toInt())
        val canvas = RgbImage.filled(W, H, config.padValue)
        ImageOps.paste(canvas, ImageOps.resizePilBilinear(crop, nw, nh), (W - nw) / 2, (H - nh) / 2)
        return canvas
    }

    fun recognize(crops: List<RgbImage>): List<Triple<String, Float, FloatArray>> {
        if (crops.isEmpty()) return emptyList()
        val t0 = System.currentTimeMillis()
        val H = config.recH; val W = config.recW; val plane = H * W
        val x = FloatArray(crops.size * 3 * plane)
        crops.forEachIndexed { k, c ->
            val lb = letterboxPlate(c)
            val base = k * 3 * plane
            for (i in 0 until plane) {
                val p = lb.argb[i]
                x[base + i] = (((p shr 16) and 0xFF) / 255f - config.mean[0]) / config.std[0]
                x[base + plane + i] = (((p shr 8) and 0xFF) / 255f - config.mean[1]) / config.std[1]
                x[base + 2 * plane + i] = ((p and 0xFF) / 255f - config.mean[2]) / config.std[2]
            }
        }
        val nl = config.letters.size
        val digits: FloatArray
        val letters: FloatArray
        OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(crops.size.toLong(), 3, H.toLong(), W.toLong())).use { input ->
            rec.run(mapOf("image" to input)).use { res ->
                digits = FloatArray(crops.size * 70).also { (res[0] as OnnxTensor).floatBuffer.get(it) }
                letters = FloatArray(crops.size * nl).also { (res[1] as OnnxTensor).floatBuffer.get(it) }
            }
        }
        val out = crops.indices.map { k ->
            val probs = FloatArray(8)
            val sb = StringBuilder()
            val digitChars = CharArray(7)
            for (slot in 0 until 7) {
                val (idx, p) = softmaxArgmax(digits, k * 70 + slot * 10, 10)
                digitChars[slot] = '0' + idx
                probs[if (slot < 2) slot else slot + 1] = p
            }
            val (li, lp) = softmaxArgmax(letters, k * nl, nl)
            probs[2] = lp
            sb.append(digitChars, 0, 2).append(config.letters[li]).append(digitChars, 2, 5)
            Triple(sb.toString(), probs.min(), probs)
        }
        lastRecognizeMs = System.currentTimeMillis() - t0
        return out
    }

    private fun softmaxArgmax(a: FloatArray, off: Int, n: Int): Pair<Int, Float> {
        var best = 0
        var mx = a[off]
        for (i in 1 until n) if (a[off + i] > mx) { mx = a[off + i]; best = i }
        var sum = 0.0
        for (i in 0 until n) sum += exp((a[off + i] - mx).toDouble())
        return best to (1.0 / sum).toFloat()
    }

    // ---------------------------------------------------------------- end to end
    fun read(img: RgbImage, conf: Float = config.conf): List<PlateResult> {
        val dets = detect(img, conf)
        val kept = ArrayList<Detection>()
        val crops = ArrayList<RgbImage>()
        for (d in dets) cropWithPadding(img, d)?.let { crops += it; kept += d }
        return recognize(crops).mapIndexed { i, (t, c, cc) -> PlateResult(kept[i], t, c, cc) }
    }

    override fun close() {
        det.close(); rec.close()
    }
}
