package ir.platereader.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import androidx.core.content.res.ResourcesCompat
import ir.platereader.core.PlateText

/** A crisp, vector-drawn Iranian licence plate: [flag strip | 12 ب 345 | ایران 67]. Always left-to-right. */
class PlateView @JvmOverloads constructor(ctx: Context, attrs: AttributeSet? = null) : View(ctx, attrs) {
    private var parts = PlateText.parts("12ب34567")
    private var raw = ""
    private val black: Typeface? = ResourcesCompat.getFont(ctx, R.font.vazirmatn_black)
    private val bold: Typeface? = ResourcesCompat.getFont(ctx, R.font.vazirmatn_bold)
    private val body = Paint(Paint.ANTI_ALIAS_FLAG)
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; color = Color.parseColor("#111111") }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#111111"); textAlign = Paint.Align.CENTER }
    private val small = Paint(Paint.ANTI_ALIAS_FLAG).apply { textAlign = Paint.Align.CENTER }
    private val r = RectF()

    fun setPlate(plate: String) {
        raw = plate
        parts = PlateText.parts(plate)
        contentDescription = PlateText.display(plate)
        invalidate()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val w = MeasureSpec.getSize(widthMeasureSpec)
        setMeasuredDimension(w, (w / 4.4f).toInt())  // real plates are 52 x 11.5 cm
    }

    override fun onDraw(c: Canvas) {
        val w = width.toFloat(); val h = height.toFloat()
        val sw = h * 0.045f
        val rad = h * 0.12f
        // body
        body.color = Color.WHITE
        r.set(sw, sw, w - sw, h - sw)
        c.drawRoundRect(r, rad, rad, body)
        // blue strip with flag
        val strip = w * 0.085f
        body.color = Color.parseColor("#1D4EA3")
        c.save(); c.clipRect(sw, sw, sw + strip, h - sw); c.drawRoundRect(r, rad, rad, body); c.restore()
        val fx = sw + strip / 2; val fw = strip * 0.55f; val fh = h * 0.16f; val fy = h * 0.42f
        val cols = intArrayOf(Color.parseColor("#239F40"), Color.WHITE, Color.parseColor("#DA0000"))
        for (i in 0..2) { body.color = cols[i]; c.drawRect(fx - fw / 2, fy + i * fh / 3, fx + fw / 2, fy + (i + 1) * fh / 3, body) }
        small.color = Color.WHITE; small.textSize = h * 0.11f; small.typeface = Typeface.DEFAULT_BOLD
        c.drawText("I.R.", fx, h * 0.72f, small); c.drawText("IRAN", fx, h * 0.84f, small)
        // region box
        val regionW = w * 0.17f
        val divX = w - sw - regionW
        stroke.strokeWidth = sw * 1.6f
        c.drawLine(divX, sw, divX, h - sw, stroke)
        small.color = Color.parseColor("#111111"); small.typeface = bold; small.textSize = h * 0.16f
        c.drawText("ایران", divX + regionW / 2, h * 0.30f, small)
        text.typeface = black
        val p = parts
        if (p != null) {
            text.textSize = h * 0.5f
            c.drawText(p.region, divX + regionW / 2, h * 0.80f, text)
            // main: first · letter · middle, laid out left to right
            val mainL = sw + strip; val mainW = divX - mainL
            text.textSize = h * 0.62f
            val base = h * 0.73f
            c.drawText(p.first, mainL + mainW * 0.19f, base, text)
            text.textSize = if (p.letter.length > 1) h * 0.42f else h * 0.56f
            c.drawText(p.letter, mainL + mainW * 0.45f, base - h * 0.02f, text)
            text.textSize = h * 0.62f
            c.drawText(p.middle, mainL + mainW * 0.76f, base, text)
        } else {
            text.textSize = h * 0.4f
            c.drawText(PlateText.toPersianDigits(raw), (sw + strip + divX) / 2, h * 0.65f, text)
        }
        stroke.strokeWidth = sw * 2
        c.drawRoundRect(r, rad, rad, stroke)
    }
}
