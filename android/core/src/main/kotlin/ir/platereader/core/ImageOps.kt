package ir.platereader.core

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/** An RGB image as packed ARGB ints (the layout of Android's Bitmap.getPixels). */
class RgbImage(val width: Int, val height: Int, val argb: IntArray) {
    init {
        require(argb.size == width * height) { "pixel buffer ${argb.size} != $width x $height" }
    }

    fun r(i: Int) = (argb[i] shr 16) and 0xFF
    fun g(i: Int) = (argb[i] shr 8) and 0xFF
    fun b(i: Int) = argb[i] and 0xFF

    companion object {
        fun filled(width: Int, height: Int, value: Int): RgbImage {
            val v = value and 0xFF
            return RgbImage(width, height, IntArray(width * height) { (0xFF shl 24) or (v shl 16) or (v shl 8) or v })
        }
    }
}

/**
 * Image operations that reproduce the Python pipeline bit-for-bit (or within 1 grey level):
 * - [resizePilBilinear]: Pillow `Image.resize(..., BILINEAR)` (antialiased, fixed-point), used by the recognizer.
 * - [resizeCvLinear]: OpenCV `cv2.resize(..., INTER_LINEAR)`, used by the YOLO letterbox.
 */
object ImageOps {
    private const val PRECISION_BITS = 32 - 8 - 2  // Pillow Resample.c, 8 bits per channel

    private fun unpack(img: RgbImage): Array<IntArray> {
        val c = Array(3) { IntArray(img.width * img.height) }
        for (i in img.argb.indices) {
            val p = img.argb[i]
            c[0][i] = (p shr 16) and 0xFF; c[1][i] = (p shr 8) and 0xFF; c[2][i] = p and 0xFF
        }
        return c
    }

    private fun pack(w: Int, h: Int, c: Array<IntArray>) =
        RgbImage(w, h, IntArray(w * h) { (0xFF shl 24) or (c[0][it] shl 16) or (c[1][it] shl 8) or c[2][it] })

    // ------------------------------------------------------------------ Pillow BILINEAR (antialiased)
    private class Coeffs(val bounds: IntArray, val kk: IntArray, val ksize: Int)

    private fun pilCoeffs(inSize: Int, outSize: Int): Coeffs {
        val scale = inSize.toDouble() / outSize
        val filterScale = max(scale, 1.0)
        val support = 1.0 * filterScale                      // bilinear filter support = 1
        val ksize = ceil(support).toInt() * 2 + 1
        val bounds = IntArray(outSize * 2)
        val kk = IntArray(outSize * ksize)
        val k = DoubleArray(ksize)
        for (xx in 0 until outSize) {
            val center = (xx + 0.5) * scale
            val ss = 1.0 / filterScale
            var xmin = (center - support + 0.5).toInt()        // C cast: truncation toward zero
            if (xmin < 0) xmin = 0
            var xmax = (center + support + 0.5).toInt()
            if (xmax > inSize) xmax = inSize
            xmax -= xmin
            var ww = 0.0
            for (x in 0 until ksize) k[x] = 0.0
            for (x in 0 until xmax) {
                var t = (x + xmin - center + 0.5) * ss
                if (t < 0) t = -t
                val w = if (t < 1.0) 1.0 - t else 0.0
                k[x] = w; ww += w
            }
            for (x in 0 until xmax) if (ww != 0.0) k[x] /= ww
            for (x in 0 until ksize) {
                val v = k[x] * (1 shl PRECISION_BITS)
                kk[xx * ksize + x] = if (k[x] < 0) (-0.5 + v).toInt() else (0.5 + v).toInt()
            }
            bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax
        }
        return Coeffs(bounds, kk, ksize)
    }

    private fun clip8(v: Int): Int = when {
        v >= (1 shl PRECISION_BITS shl 8) -> 255
        v <= 0 -> 0
        else -> v shr PRECISION_BITS
    }

    /** Pillow-exact `Image.resize((w, h), Image.BILINEAR)`: horizontal pass then vertical pass, 8-bit in between. */
    fun resizePilBilinear(src: RgbImage, outW: Int, outH: Int): RgbImage {
        if (outW == src.width && outH == src.height) return RgbImage(outW, outH, src.argb.copyOf())
        var c = unpack(src)
        var w = src.width
        val h = src.height
        if (outW != w) {
            val cf = pilCoeffs(w, outW)
            c = Array(3) { ch ->
                val inp = c[ch]
                IntArray(outW * h).also { out ->
                    for (y in 0 until h) for (xx in 0 until outW) {
                        val xmin = cf.bounds[xx * 2]; val xmax = cf.bounds[xx * 2 + 1]
                        var ss = 1 shl (PRECISION_BITS - 1)
                        for (x in 0 until xmax) ss += inp[y * w + x + xmin] * cf.kk[xx * cf.ksize + x]
                        out[y * outW + xx] = clip8(ss)
                    }
                }
            }
            w = outW
        }
        if (outH != h) {
            val cf = pilCoeffs(h, outH)
            c = Array(3) { ch ->
                val inp = c[ch]
                IntArray(w * outH).also { out ->
                    for (yy in 0 until outH) {
                        val ymin = cf.bounds[yy * 2]; val ymax = cf.bounds[yy * 2 + 1]
                        for (x in 0 until w) {
                            var ss = 1 shl (PRECISION_BITS - 1)
                            for (y in 0 until ymax) ss += inp[(y + ymin) * w + x] * cf.kk[yy * cf.ksize + y]
                            out[yy * w + x] = clip8(ss)
                        }
                    }
                }
            }
        }
        return pack(w, outH, c)
    }

    // ------------------------------------------------------------------ OpenCV INTER_LINEAR (fixed point)
    private const val COEF_BITS = 11
    private const val COEF_SCALE = 1 shl COEF_BITS

    private fun cvTaps(inSize: Int, outSize: Int): Pair<IntArray, IntArray> {
        val scale = inSize.toDouble() / outSize
        val idx = IntArray(outSize)
        val alpha = IntArray(outSize)  // weight of idx+1, in 1/2048
        for (d in 0 until outSize) {
            var f = ((d + 0.5) * scale - 0.5).toFloat()
            var s = floor(f).toInt()
            f -= s
            if (s < 0) { f = 0f; s = 0 }
            if (s >= inSize - 1) { f = 0f; s = inSize - 1 }
            idx[d] = s
            alpha[d] = Math.round(f * COEF_SCALE)
        }
        return idx to alpha
    }

    /** OpenCV-compatible `cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)`. */
    fun resizeCvLinear(src: RgbImage, outW: Int, outH: Int): RgbImage {
        if (outW == src.width && outH == src.height) return RgbImage(outW, outH, src.argb.copyOf())
        val c = unpack(src)
        val (xi, xa) = cvTaps(src.width, outW)
        val (yi, ya) = cvTaps(src.height, outH)
        val w = src.width
        val out = Array(3) { IntArray(outW * outH) }
        val row0 = IntArray(outW)
        val row1 = IntArray(outW)
        for (ch in 0 until 3) {
            val inp = c[ch]
            for (y in 0 until outH) {
                val sy = yi[y]; val sy1 = min(sy + 1, src.height - 1)
                for (x in 0 until outW) {
                    val sx = xi[x]; val sx1 = min(sx + 1, w - 1); val a = xa[x]
                    row0[x] = inp[sy * w + sx] * (COEF_SCALE - a) + inp[sy * w + sx1] * a
                    row1[x] = inp[sy1 * w + sx] * (COEF_SCALE - a) + inp[sy1 * w + sx1] * a
                }
                val b = ya[y]
                for (x in 0 until outW) {
                    val v = (row0[x].toLong() * (COEF_SCALE - b) + row1[x].toLong() * b + (1L shl (2 * COEF_BITS - 1))) shr (2 * COEF_BITS)
                    out[ch][y * outW + x] = v.toInt().coerceIn(0, 255)
                }
            }
        }
        return pack(outW, outH, out)
    }

    /** Crop [x1, x2) x [y1, y2) (already clamped integer pixel box). */
    fun crop(src: RgbImage, x1: Int, y1: Int, x2: Int, y2: Int): RgbImage {
        val w = x2 - x1; val h = y2 - y1
        val out = IntArray(w * h)
        for (y in 0 until h) System.arraycopy(src.argb, (y + y1) * src.width + x1, out, y * w, w)
        return RgbImage(w, h, out)
    }

    /** Paste [src] into [dst] at (left, top). */
    fun paste(dst: RgbImage, src: RgbImage, left: Int, top: Int) {
        for (y in 0 until src.height) System.arraycopy(src.argb, y * src.width, dst.argb, (y + top) * dst.width + left, src.width)
    }
}
