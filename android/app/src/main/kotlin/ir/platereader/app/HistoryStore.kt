package ir.platereader.app

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.core.content.FileProvider
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

data class HistoryItem(val id: Long, val time: Long, val text: String, val detConf: Float, val textConf: Float, val cropPath: String)

/** Saved reads: a small JSON index + one JPEG crop per plate, in the app's private storage. */
class HistoryStore(private val ctx: Context) {
    private val dir = File(ctx.filesDir, "history").apply { mkdirs() }
    private val index = File(dir, "index.json")

    @Synchronized
    fun all(): List<HistoryItem> {
        if (!index.exists()) return emptyList()
        val arr = JSONArray(index.readText())
        return List(arr.length()) { i ->
            arr.getJSONObject(i).run {
                HistoryItem(getLong("id"), getLong("time"), getString("text"), getDouble("det").toFloat(),
                            getDouble("conf").toFloat(), getString("crop"))
            }
        }.sortedByDescending { it.time }
    }

    @Synchronized
    fun add(text: String, detConf: Float, textConf: Float, crop: Bitmap): HistoryItem {
        val now = System.currentTimeMillis()
        // the same plate read again within a minute (re-scan of the same car) is not saved twice
        all().firstOrNull { it.text == text && now - it.time < 60_000 }?.let { return it }
        val file = File(dir, "crop_$now${(0..999).random()}.jpg")
        file.outputStream().use { crop.compress(Bitmap.CompressFormat.JPEG, 90, it) }
        val item = HistoryItem(now, now, text, detConf, textConf, file.absolutePath)
        write(all() + item)
        return item
    }

    @Synchronized
    fun delete(id: Long) {
        val items = all()
        items.firstOrNull { it.id == id }?.let { File(it.cropPath).delete() }
        write(items.filter { it.id != id })
    }

    @Synchronized
    fun clear() {
        all().forEach { File(it.cropPath).delete() }
        write(emptyList())
    }

    private fun write(items: List<HistoryItem>) {
        val arr = JSONArray()
        items.forEach {
            arr.put(JSONObject().put("id", it.id).put("time", it.time).put("text", it.text).put("det", it.detConf.toDouble())
                        .put("conf", it.textConf.toDouble()).put("crop", it.cropPath))
        }
        index.writeText(arr.toString())
    }

    /** CSV (UTF-8 with BOM so Excel shows Persian) shared through a FileProvider. */
    fun exportCsv(latin: (String) -> String): Uri {
        val out = File(ctx.cacheDir, "exports").apply { mkdirs() }.resolve("plates.csv")
        out.writeText("﻿plate,latin,date_jalali,timestamp\n" +
            all().joinToString("\n") { "${it.text},${latin(it.text)},${Fmt.date(it.time)},${it.time}" })
        return FileProvider.getUriForFile(ctx, "${ctx.packageName}.files", out)
    }
}
