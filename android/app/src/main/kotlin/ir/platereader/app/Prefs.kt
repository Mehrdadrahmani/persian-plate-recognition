package ir.platereader.app

import android.content.Context

/** User settings (SharedPreferences). */
class Prefs(ctx: Context) {
    private val sp = ctx.getSharedPreferences("settings", Context.MODE_PRIVATE)

    var onboarded: Boolean
        get() = sp.getBoolean("onboarded", false)
        set(v) = sp.edit().putBoolean("onboarded", v).apply()

    /** 0..100, 50 = the threshold chosen on the validation split. */
    var sensitivity: Int
        get() = sp.getInt("sensitivity", 50)
        set(v) = sp.edit().putInt("sensitivity", v).apply()

    var autoSave: Boolean
        get() = sp.getBoolean("autosave", true)
        set(v) = sp.edit().putBoolean("autosave", v).apply()

    var vibrate: Boolean
        get() = sp.getBoolean("vibrate", true)
        set(v) = sp.edit().putBoolean("vibrate", v).apply()

    /** Detector threshold from the sensitivity slider: 50 -> base, 100 -> base - 0.25, 0 -> base + 0.25. */
    fun threshold(base: Float): Float = (base + (50 - sensitivity) / 200f).coerceIn(0.2f, 0.9f)
}
