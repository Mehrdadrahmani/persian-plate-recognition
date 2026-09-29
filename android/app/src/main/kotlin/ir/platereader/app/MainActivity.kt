package ir.platereader.app

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import ir.platereader.app.databinding.ActivityMainBinding
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    private lateinit var ui: ActivityMainBinding
    private val tags = mapOf(R.id.nav_scan to "scan", R.id.nav_history to "history", R.id.nav_settings to "settings")

    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        if (!Prefs(this).onboarded) {
            startActivity(Intent(this, IntroActivity::class.java))
            finish()
            return
        }
        ui = ActivityMainBinding.inflate(layoutInflater)
        setContentView(ui.root)
        lifecycleScope.launch { Engines.get(this@MainActivity) }  // warm up the models while the camera starts
        ui.bottomNav.setOnItemSelectedListener { show(it.itemId); true }
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(ui.bottomNav) { v, insets ->
            val nav = insets.getInsets(androidx.core.view.WindowInsetsCompat.Type.navigationBars())
            v.setPadding(0, 0, 0, nav.bottom); insets
        }
        if (savedInstanceState == null) show(R.id.nav_scan)
    }

    fun openHistory() {
        ui.bottomNav.selectedItemId = R.id.nav_history
    }

    private fun show(id: Int) {
        val fm = supportFragmentManager
        val tx = fm.beginTransaction().setReorderingAllowed(true)
        tags.forEach { (itemId, tag) ->
            val f = fm.findFragmentByTag(tag)
            if (itemId == id) {
                if (f == null) tx.add(R.id.container, create(itemId), tag) else tx.show(f)
            } else if (f != null) tx.hide(f)
        }
        tx.commitNow()
        (fm.findFragmentByTag("history") as? HistoryFragment)?.takeIf { id == R.id.nav_history }?.refresh()
    }

    private fun create(id: Int): Fragment = when (id) {
        R.id.nav_history -> HistoryFragment()
        R.id.nav_settings -> SettingsFragment()
        else -> ScanFragment()
    }
}
