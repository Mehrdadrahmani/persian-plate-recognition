package ir.platereader.app

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.RectF
import ir.platereader.core.PlateResult
import ir.platereader.core.RgbImage
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

object ImageUtils {
    /** Downscale so the longer side is at most [maxSide] (the detector works at 640 px anyway). */
    fun limit(b: Bitmap, maxSide: Int = 1920): Bitmap {
        val s = maxSide.toFloat() / max(b.width, b.height)
        return if (s >= 1f) b else Bitmap.createScaledBitmap(b, (b.width * s).roundToInt(), (b.height * s).roundToInt(), true)
    }

    fun rotate(b: Bitmap, degrees: Int): Bitmap =
        if (degrees == 0) b else Bitmap.createBitmap(b, 0, 0, b.width, b.height, Matrix().apply { postRotate(degrees.toFloat()) }, true)

    fun toRgb(b: Bitmap): RgbImage {
        val bmp = if (b.config == Bitmap.Config.ARGB_8888) b else b.copy(Bitmap.Config.ARGB_8888, false)
        val px = IntArray(bmp.width * bmp.height)
        bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        return RgbImage(bmp.width, bmp.height, px)
    }

    /** Plate crop (box enlarged by 12%) for the result cards and history thumbnails. */
    fun crop(b: Bitmap, r: PlateResult): Bitmap {
        val d = r.box; val pw = (d.x2 - d.x1) * 0.12f; val ph = (d.y2 - d.y1) * 0.25f
        val x1 = max(0, (d.x1 - pw).roundToInt()); val y1 = max(0, (d.y1 - ph).roundToInt())
        val x2 = min(b.width, (d.x2 + pw).roundToInt()); val y2 = min(b.height, (d.y2 + ph).roundToInt())
        return Bitmap.createBitmap(b, x1, y1, max(1, x2 - x1), max(1, y2 - y1))
    }

    fun annotate(src: Bitmap, results: List<PlateResult>, drawing: PlateDrawing): Bitmap {
        val out = src.copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(out)
        val scale = max(out.width, out.height) / 520f
        results.forEach { drawing.draw(c, RectF(it.box.x1, it.box.y1, it.box.x2, it.box.y2), it.text, it.textConf, scale, out.width.toFloat()) }
        return out
    }
}
