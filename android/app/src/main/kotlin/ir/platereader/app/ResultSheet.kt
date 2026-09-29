package ir.platereader.app

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.google.android.material.button.MaterialButton
import ir.platereader.app.databinding.SheetResultBinding
import ir.platereader.core.PlateResult
import ir.platereader.core.PlateText

/** Hands the latest result to the sheet (bitmaps are too large for fragment arguments). */
object ResultStore {
    var image: Bitmap? = null; private set
    var annotated: Bitmap? = null; private set
    var results: List<PlateResult> = emptyList(); private set
    var ms = 0L; private set

    fun set(img: Bitmap, r: List<PlateResult>, time: Long, ann: Bitmap) {
        image = img; results = r; ms = time; annotated = ann
    }
}

class ResultSheet : BottomSheetDialogFragment() {
    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, s: Bundle?): View {
        val ui = SheetResultBinding.inflate(inflater, container, false)
        val ctx = requireContext()
        val results = ResultStore.results
        val img = ResultStore.image
        ui.image.setImageBitmap(ResultStore.annotated)
        ui.count.text = getString(R.string.result_count, Fmt.fa(results.size))
        ui.timeChip.text = getString(R.string.timing, Fmt.fa(ResultStore.ms))
        ui.timeChip.visibility = if (ResultStore.ms > 0) View.VISIBLE else View.GONE
        ui.empty.visibility = if (results.isEmpty()) View.VISIBLE else View.GONE
        val cfg = Engines.peek()?.config
        for (r in results) {
            val v = inflater.inflate(R.layout.item_result_plate, ui.plates, false)
            v.findViewById<PlateView>(R.id.plate).setPlate(r.text)
            if (img != null) v.findViewById<ImageView>(R.id.crop).setImageBitmap(ImageUtils.crop(img, r))
            v.findViewById<TextView>(R.id.latin).text = cfg?.let { PlateText.latin(r.text, it.letters, it.latin) } ?: ""
            v.findViewById<View>(R.id.warn).visibility = if (r.textConf < 0.5f) View.VISIBLE else View.GONE
            v.findViewById<MaterialButton>(R.id.copy).setOnClickListener {
                (ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager).setPrimaryClip(ClipData.newPlainText("plate", r.text))
                Toast.makeText(ctx, R.string.copied, Toast.LENGTH_SHORT).show()
            }
            v.findViewById<MaterialButton>(R.id.share).setOnClickListener {
                startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).setType("text/plain")
                    .putExtra(Intent.EXTRA_TEXT, getString(R.string.share_text, PlateText.display(r.text))), getString(R.string.share)))
            }
            ui.plates.addView(v)
        }
        return ui.root
    }

    override fun onStart() {
        super.onStart()
        (dialog as? BottomSheetDialog)?.behavior?.apply { state = BottomSheetBehavior.STATE_EXPANDED; skipCollapsed = true }
    }
}
