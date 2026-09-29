package ir.platereader.app

import android.content.Context
import ai.onnxruntime.OrtEnvironment
import ir.platereader.core.PlateConfig
import ir.platereader.core.PlateEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** One shared plate engine (INT8 mobile models), loaded once in the background. */
object Engines {
    @Volatile private var engine: PlateEngine? = null
    private val lock = Mutex()

    fun peek(): PlateEngine? = engine

    suspend fun get(ctx: Context): PlateEngine = engine ?: lock.withLock {
        engine ?: withContext(Dispatchers.Default) { load(ctx.applicationContext) }.also { engine = it }
    }

    private fun load(ctx: Context): PlateEngine {
        val a = ctx.assets
        val cfg = PlateConfig.parse(a.open("pipeline.properties").bufferedReader().readText(),
                                    a.open("letters.tsv").bufferedReader(Charsets.UTF_8).readText())
        val det = a.open("lpd_mobile.onnx").readBytes()
        val rec = a.open("lpr_mobile.onnx").readBytes()
        val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
        val env = OrtEnvironment.getEnvironment()
        return PlateEngine(env, det, rec, cfg, threads = threads)
    }
}
