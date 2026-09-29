package ir.platereader.app

import android.content.Intent
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import ir.platereader.app.databinding.ActivityIntroBinding

class IntroActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val ui = ActivityIntroBinding.inflate(layoutInflater)
        setContentView(ui.root)
        listOf(Triple(ui.f1.root, R.drawable.ic_bolt, R.string.intro_f1_t to R.string.intro_f1_d),
               Triple(ui.f2.root, R.drawable.ic_live, R.string.intro_f2_t to R.string.intro_f2_d),
               Triple(ui.f3.root, R.drawable.ic_shield, R.string.intro_f3_t to R.string.intro_f3_d)).forEach { (row, icon, texts) ->
            row.findViewById<ImageView>(R.id.icon).setImageResource(icon)
            row.findViewById<TextView>(R.id.title).setText(texts.first)
            row.findViewById<TextView>(R.id.desc).setText(texts.second)
        }
        val pad = ui.bottom.paddingBottom
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(ui.bottom) { v, insets ->
            val nav = insets.getInsets(androidx.core.view.WindowInsetsCompat.Type.navigationBars())
            v.setPadding(v.paddingLeft, v.paddingTop, v.paddingRight, pad + nav.bottom)
            insets
        }
        ui.start.setOnClickListener {
            Prefs(this).onboarded = true
            startActivity(Intent(this, MainActivity::class.java))
            finish()
        }
    }
}
