package ir.platereader.app

import ir.platereader.core.PlateResult

/** Live mode: a plate is "confirmed" once the same text is read in 2 frames; each plate is reported once. */
class LiveTracker(private val minHits: Int = 2, private val minConf: Float = 0.5f) {
    private val hits = HashMap<String, Int>()
    private val best = HashMap<String, PlateResult>()
    private val confirmed = LinkedHashSet<String>()

    fun update(results: List<PlateResult>): List<PlateResult> {
        val newly = ArrayList<PlateResult>()
        for (r in results) {
            if (r.textConf < minConf) continue
            hits[r.text] = (hits[r.text] ?: 0) + 1
            if ((best[r.text]?.textConf ?: 0f) < r.textConf) best[r.text] = r
            if (hits.getValue(r.text) >= minHits && confirmed.add(r.text)) newly += best.getValue(r.text)
        }
        return newly
    }

    val count get() = confirmed.size

    fun reset() {
        hits.clear(); best.clear(); confirmed.clear()
    }
}
