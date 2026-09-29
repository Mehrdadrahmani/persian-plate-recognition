package ir.platereader.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import androidx.core.content.ContextCompat
import androidx.core.content.res.ResourcesCompat
import ir.platereader.core.PlateText
import kotlin.math.max

/** Draws a detection box + a label with the plate in printed (left-to-right) order, piece by piece. */
class PlateDrawing(ctx: Context) {
    private val app = ctx.applicationContext
    val box = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val bg = Paint(Paint.ANTI_ALIAS_FLAG)
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; typeface = ResourcesCompat.getFont(ctx, R.font.vazirmatn_black)
    }

    fun draw(c: Canvas, rect: RectF, plate: String, conf: Float, scale: Float, imageW: Float) {
        val color = ContextCompat.getColor(app, R.color.brand_amber)
        box.color = color; box.strokeWidth = max(3f, 3.5f * scale)
        c.drawRoundRect(rect, 6f * scale, 6f * scale, box)
        val p = PlateText.parts(plate)
        val pieces = if (p != null) listOf(p.first, " ", p.letter, " ", p.middle, "  ", p.region) else listOf(PlateText.toPersianDigits(plate))
        text.textSize = 15f * scale
        val pad = 6f * scale
        val w = pieces.sumOf { text.measureText(it).toDouble() }.toFloat() + 2 * pad
        val fm = text.fontMetrics
        val h = fm.descent - fm.ascent + pad
        val top = if (rect.top - h - 4 * scale > 0) rect.top - h - 4 * scale else rect.bottom + 4 * scale
        val left = rect.left.coerceIn(0f, max(0f, imageW - w))
        bg.color = color
        c.drawRoundRect(RectF(left, top, left + w, top + h), 8f * scale, 8f * scale, bg)
        var x = left + pad
        for (s in pieces) { c.drawText(s, x, top - fm.ascent + pad / 2, text); x += text.measureText(s) }
    }
}
