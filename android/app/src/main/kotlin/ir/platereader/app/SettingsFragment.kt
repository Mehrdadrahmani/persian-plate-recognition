package ir.platereader.app

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.fragment.app.Fragment
import ir.platereader.app.databinding.FragmentSettingsBinding

class SettingsFragment : Fragment() {
    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, s: Bundle?): View {
        val ui = FragmentSettingsBinding.inflate(inflater, container, false)
        val prefs = Prefs(requireContext())
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(ui.root) { v, insets ->
            v.setPadding(0, insets.getInsets(androidx.core.view.WindowInsetsCompat.Type.statusBars()).top, 0, 0); insets
        }
        ui.sensitivity.value = prefs.sensitivity.toFloat()
        ui.sensitivity.addOnChangeListener { _, v, _ -> prefs.sensitivity = v.toInt() }
        ui.sensitivity.setLabelFormatter { Fmt.fa(it.toInt()) }
        ui.autosave.isChecked = prefs.autoSave
        ui.autosave.setOnCheckedChangeListener { _, b -> prefs.autoSave = b }
        ui.vibrate.isChecked = prefs.vibrate
        ui.vibrate.setOnCheckedChangeListener { _, b -> prefs.vibrate = b }
        fun row(root: View, icon: Int, title: String, desc: String) {
            root.findViewById<ImageView>(R.id.icon).setImageResource(icon)
            root.findViewById<TextView>(R.id.title).text = title
            root.findViewById<TextView>(R.id.desc).text = desc
        }
        row(ui.aboutPrivacy.root, R.drawable.ic_shield, getString(R.string.s_privacy_t), getString(R.string.s_privacy_d))
        row(ui.aboutModels.root, R.drawable.ic_chip, getString(R.string.s_models_t), getString(R.string.s_models_d))
        row(ui.aboutEngine.root, R.drawable.ic_bolt, getString(R.string.s_speed_t), getString(R.string.s_speed_d))
        ui.version.text = getString(R.string.s_version, ir.platereader.core.PlateText.toPersianDigits(BuildConfig.VERSION_NAME))
        return ui.root
    }
}
