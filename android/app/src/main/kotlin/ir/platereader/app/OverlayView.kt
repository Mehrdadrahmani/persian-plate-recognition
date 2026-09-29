package ir.platereader.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import ir.platereader.core.PlateResult
import kotlin.math.max

/** Camera overlay: a scanning frame (photo mode) or live detection boxes mapped onto the preview (live mode). */
class OverlayView @JvmOverloads constructor(ctx: Context, attrs: AttributeSet? = null) : View(ctx, attrs) {
    private val drawing = PlateDrawing(ctx)
    private val scrim = Paint().apply { color = Color.parseColor("#59000000") }
    private val corner = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; color = Color.WHITE; strokeWidth = 4 * resources.displayMetrics.density; strokeCap = Paint.Cap.ROUND
    }
    private val frame = RectF()
    private val path = Path()
    private var results: List<PlateResult> = emptyList()
    private var imgW = 1; private var imgH = 1
    var live = false
        set(v) { field = v; if (!v) results = emptyList(); invalidate() }

    fun setResults(r: List<PlateResult>, w: Int, h: Int) {
        results = r; imgW = w; imgH = h; postInvalidate()
    }

    override fun onDraw(c: Canvas) {
        val d = resources.displayMetrics.density
        if (!live) {
            val fw = width * 0.84f; val fh = fw / 2.6f
            frame.set((width - fw) / 2, height * 0.40f - fh / 2, (width + fw) / 2, height * 0.40f + fh / 2)
            path.reset(); path.addRect(0f, 0f, width.toFloat(), height.toFloat(), Path.Direction.CW)
            path.addRoundRect(frame, 22 * d, 22 * d, Path.Direction.CCW)
            c.drawPath(path, scrim)
            val L = 34 * d
            with(frame) {
                c.drawLine(left, top + L, left, top + 10 * d, corner); c.drawLine(left + 10 * d, top, left + L, top, corner)
                c.drawLine(right - L, top, right - 10 * d, top, corner); c.drawLine(right, top + 10 * d, right, top + L, corner)
                c.drawLine(left, bottom - L, left, bottom - 10 * d, corner); c.drawLine(left + 10 * d, bottom, left + L, bottom, corner)
                c.drawLine(right - L, bottom, right - 10 * d, bottom, corner); c.drawLine(right, bottom - L, right, bottom - 10 * d, corner)
            }
            return
        }
        // PreviewView FILL_CENTER: same scale on both axes, centre crop
        val s = max(width.toFloat() / imgW, height.toFloat() / imgH)
        val ox = (width - imgW * s) / 2; val oy = (height - imgH * s) / 2
        for (r in results) {
            val b = r.box
            drawing.draw(c, RectF(ox + b.x1 * s, oy + b.y1 * s, ox + b.x2 * s, oy + b.y2 * s), r.text, r.textConf, d, width.toFloat())
        }
    }
}
