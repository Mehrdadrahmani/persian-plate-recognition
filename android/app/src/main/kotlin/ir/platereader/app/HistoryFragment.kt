package ir.platereader.app

import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.widget.doAfterTextChanged
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.ItemTouchHelper
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import ir.platereader.app.databinding.FragmentHistoryBinding
import ir.platereader.app.databinding.ItemHistoryBinding
import ir.platereader.core.PlateText

class HistoryFragment : Fragment() {
    private var _ui: FragmentHistoryBinding? = null
    private val ui get() = _ui!!
    private lateinit var store: HistoryStore
    private val adapter = Adapter()
    private var all: List<HistoryItem> = emptyList()

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, s: Bundle?): View {
        _ui = FragmentHistoryBinding.inflate(inflater, container, false)
        return ui.root
    }

    override fun onViewCreated(view: View, s: Bundle?) {
        store = HistoryStore(requireContext())
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(ui.root) { v, insets ->
            v.setPadding(0, insets.getInsets(androidx.core.view.WindowInsetsCompat.Type.statusBars()).top, 0, 0); insets
        }
        ui.list.layoutManager = LinearLayoutManager(requireContext())
        ui.list.adapter = adapter
        ui.search.doAfterTextChanged { filter() }
        ui.export.setOnClickListener { export() }
        ui.clear.setOnClickListener {
            MaterialAlertDialogBuilder(requireContext()).setMessage(R.string.clear_confirm)
                .setNegativeButton(R.string.cancel, null)
                .setPositiveButton(R.string.delete) { _, _ -> store.clear(); refresh() }.show()
        }
        ItemTouchHelper(object : ItemTouchHelper.SimpleCallback(0, ItemTouchHelper.LEFT or ItemTouchHelper.RIGHT) {
            override fun onMove(rv: RecyclerView, a: RecyclerView.ViewHolder, b: RecyclerView.ViewHolder) = false
            override fun onSwiped(vh: RecyclerView.ViewHolder, dir: Int) {
                store.delete(adapter.items[vh.bindingAdapterPosition].id); refresh()
            }
        }).attachToRecyclerView(ui.list)
        refresh()
    }

    override fun onDestroyView() {
        _ui = null
        super.onDestroyView()
    }

    fun refresh() {
        if (_ui == null) return
        all = store.all()
        ui.count.text = getString(R.string.history_count, Fmt.fa(all.size))
        filter()
    }

    private fun filter() {
        val q = Fmt.latinDigits(ui.search.text?.toString().orEmpty()).replace(" ", "")
        adapter.items = if (q.isEmpty()) all else all.filter { Fmt.latinDigits(it.text).contains(q) }
        adapter.notifyDataSetChanged()
        ui.empty.visibility = if (adapter.items.isEmpty()) View.VISIBLE else View.GONE
    }

    private fun export() {
        val cfg = Engines.peek()?.config
        val uri = store.exportCsv { t -> cfg?.let { PlateText.latin(t, it.letters, it.latin) } ?: t }
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).setType("text/csv")
            .putExtra(Intent.EXTRA_STREAM, uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION), getString(R.string.export_csv)))
    }

    private inner class Adapter : RecyclerView.Adapter<Adapter.VH>() {
        var items: List<HistoryItem> = emptyList()

        inner class VH(val b: ItemHistoryBinding) : RecyclerView.ViewHolder(b.root)

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
            VH(ItemHistoryBinding.inflate(LayoutInflater.from(parent.context), parent, false))

        override fun getItemCount() = items.size

        override fun onBindViewHolder(h: VH, i: Int) {
            val it = items[i]
            h.b.plate.setPlate(it.text)
            h.b.date.text = Fmt.date(it.time)
            h.b.crop.setImageBitmap(BitmapFactory.decodeFile(it.cropPath))
        }
    }
}
