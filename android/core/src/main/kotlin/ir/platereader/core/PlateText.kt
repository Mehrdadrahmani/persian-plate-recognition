package ir.platereader.core

/** Display parts of an Iranian plate: two digits, letter, three digits, region code (Persian digits). */
data class PlateParts(val first: String, val letter: String, val middle: String, val region: String)

object PlateText {
    private val PERSIAN_DIGITS = charArrayOf('۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹')
    private val PLATE_RE = Regex("^(\\d{2})(\\D+)(\\d{5})$")

    fun toPersianDigits(s: String): String =
        buildString { for (ch in s) append(if (ch in '0'..'9') PERSIAN_DIGITS[ch - '0'] else ch) }

    /** '12ب34567' -> PlateParts('۱۲', 'ب', '۳۴۵', '۶۷'); null if the text is not a plate. */
    fun parts(text: String): PlateParts? {
        val m = PLATE_RE.matchEntire(text) ?: return null
        val (a, letter, rest) = m.destructured
        return PlateParts(toPersianDigits(a), letter, toPersianDigits(rest.substring(0, 3)), toPersianDigits(rest.substring(3)))
    }

    /** '12ب34567' -> '12-B-345-67' using the Latin names from the config. */
    fun latin(text: String, letters: List<String>, latin: List<String>): String {
        val m = PLATE_RE.matchEntire(text) ?: return text
        val (a, letter, rest) = m.destructured
        val name = letters.indexOf(letter).let { if (it >= 0) latin[it] else "?" }
        return "$a-$name-${rest.substring(0, 3)}-${rest.substring(3)}"
    }

    /** Persian plate in its printed left-to-right order, e.g. '۱۲ ب ۳۴۵ | ۶۷'. */
    fun display(text: String): String = parts(text)?.let { "${it.first} ${it.letter} ${it.middle} | ${it.region}" } ?: text
}
