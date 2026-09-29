package ir.platereader.app

import android.content.Context
import androidx.core.content.ContextCompat
import ir.platereader.core.PlateText
import java.util.Calendar
import kotlin.math.roundToInt

/** Persian formatting helpers: digits, percentages, Jalali (Shamsi) dates, confidence colours. */
object Fmt {
    fun fa(n: Number): String = PlateText.toPersianDigits(n.toString())

    fun pct(v: Float): String = fa((v * 100).roundToInt()) + "٪"

    fun confColor(ctx: Context, v: Float): Int = ContextCompat.getColor(
        ctx, when {
            v >= 0.8f -> R.color.conf_high
            v >= 0.5f -> R.color.conf_mid
            else -> R.color.conf_low
        })

    /** Gregorian -> Jalali (algorithm of jdf / jalaali). */
    fun toJalali(gy: Int, gm: Int, gd: Int): Triple<Int, Int, Int> {
        val gdm = intArrayOf(0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
        val gy2 = if (gm > 2) gy + 1 else gy
        var days = 355666 + 365 * gy + (gy2 + 3) / 4 - (gy2 + 99) / 100 + (gy2 + 399) / 400 + gd + gdm[gm - 1]
        var jy = -1595 + 33 * (days / 12053)
        days %= 12053
        jy += 4 * (days / 1461)
        days %= 1461
        if (days > 365) {
            jy += (days - 1) / 365
            days = (days - 1) % 365
        }
        val jm = if (days < 186) 1 + days / 31 else 7 + (days - 186) / 30
        val jd = 1 + if (days < 186) days % 31 else (days - 186) % 30
        return Triple(jy, jm, jd)
    }

    private val MONTHS = arrayOf("فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند")

    /** e.g. "۵ مهر ۱۴۰۵ · ۱۴:۳۲" */
    fun date(millis: Long): String {
        val c = Calendar.getInstance().apply { timeInMillis = millis }
        val (y, m, d) = toJalali(c.get(Calendar.YEAR), c.get(Calendar.MONTH) + 1, c.get(Calendar.DAY_OF_MONTH))
        val hh = c.get(Calendar.HOUR_OF_DAY).toString().padStart(2, '0')
        val mm = c.get(Calendar.MINUTE).toString().padStart(2, '0')
        return PlateText.toPersianDigits("$d ${MONTHS[m - 1]} $y · $hh:$mm")
    }

    /** Persian or Latin digits -> Latin digits (for search). */
    fun latinDigits(s: String): String =
        buildString { for (ch in s) append(if (ch in '۰'..'۹') '0' + (ch - '۰') else if (ch in '٠'..'٩') '0' + (ch - '٠') else ch) }
}
