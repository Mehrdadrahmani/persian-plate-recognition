package ir.platereader.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.provider.MediaStore
import android.util.Size
import android.view.HapticFeedbackConstants
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.google.android.material.chip.Chip
import ir.platereader.app.databinding.FragmentScanBinding
import ir.platereader.core.PlateEngine
import ir.platereader.core.PlateResult
import ir.platereader.core.PlateText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class ScanFragment : Fragment() {
    private var _ui: FragmentScanBinding? = null
    private val ui get() = _ui!!
    private lateinit var prefs: Prefs
    private lateinit var history: HistoryStore
    private var engine: PlateEngine? = null
    private var camera: Camera? = null
    private var capture: ImageCapture? = null
    private var provider: ProcessCameraProvider? = null
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private val analyzing = AtomicBoolean(false)
    private val tracker = LiveTracker()
    private var torch = false
    private var busy = false

    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted -> onPermission(granted) }
    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let { u -> viewLifecycleOwner.lifecycleScope.launch { decode(u)?.let { processPhoto(it) } } }
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, s: Bundle?): View {
        _ui = FragmentScanBinding.inflate(inflater, container, false)
        return ui.root
    }

    override fun onViewCreated(view: View, s: Bundle?) {
        prefs = Prefs(requireContext())
        history = HistoryStore(requireContext())
        val pad = ui.topBar.paddingTop
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(ui.topBar) { v, insets ->  // edge-to-edge (Android 15)
            val top = insets.getInsets(androidx.core.view.WindowInsetsCompat.Type.statusBars()).top
            v.setPadding(v.paddingLeft, pad + top, v.paddingRight, v.paddingBottom); insets
        }
        ui.shutter.setOnClickListener { takePhoto() }
        ui.gallery.setOnClickListener { openGallery() }
        ui.galleryAlt.setOnClickListener { openGallery() }
        ui.grant.setOnClickListener { permission.launch(Manifest.permission.CAMERA) }
        ui.lastThumb.setOnClickListener { (activity as? MainActivity)?.openHistory() }
        ui.flash.setOnClickListener { toggleTorch() }
        ui.liveChip.setOnCheckedChangeListener { _, on -> setLive(on) }
        ui.shutter.isEnabled = false
        viewLifecycleOwner.lifecycleScope.launch {
            engine = Engines.get(requireContext())
            ui.shutter.isEnabled = true
            ui.hint.setText(if (ui.liveChip.isChecked) R.string.scan_live_hint else R.string.scan_hint)
        }
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) onPermission(true)
        else permission.launch(Manifest.permission.CAMERA)
    }

    override fun onHiddenChanged(hidden: Boolean) {  // stop the camera while another tab is shown
        if (hidden) provider?.unbindAll() else if (_ui != null && hasCamera()) bindCamera()
    }

    override fun onDestroyView() {
        provider?.unbindAll()
        _ui = null
        super.onDestroyView()
    }

    override fun onDestroy() {
        analysisExecutor.shutdown()
        super.onDestroy()
    }

    // ---------------------------------------------------------------- camera
    private fun hasCamera() = ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun onPermission(granted: Boolean) {
        ui.permissionPanel.visibility = if (granted) View.GONE else View.VISIBLE
        if (granted) bindCamera()
    }

    private fun selector(w: Int, h: Int) = ResolutionSelector.Builder()
        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
        .setResolutionStrategy(ResolutionStrategy(Size(w, h), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
        .build()

    private fun bindCamera() {
        val future = ProcessCameraProvider.getInstance(requireContext())
        future.addListener({
            val p = future.get().also { provider = it }
            if (_ui == null || isHidden) return@addListener
            val preview = Preview.Builder().setResolutionSelector(selector(1280, 960)).build()
                .also { it.surfaceProvider = ui.preview.surfaceProvider }
            // ~2 MP is plenty: the detector looks at 640 px, crops come from this image
            capture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setResolutionSelector(selector(1920, 1440)).build()
            val useCases = mutableListOf(preview, capture!!)
            if (ui.liveChip.isChecked) {
                useCases += ImageAnalysis.Builder().setResolutionSelector(selector(1280, 960))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).build()
                    .also { it.setAnalyzer(analysisExecutor, ::analyze) }
            }
            try {
                p.unbindAll()
                camera = p.bindToLifecycle(viewLifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, *useCases.toTypedArray())
                camera?.cameraControl?.enableTorch(torch)
            } catch (e: Exception) {
                ui.hint.text = getString(R.string.error, e.message)
            }
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    private fun toggleTorch() {
        torch = !torch
        camera?.cameraControl?.enableTorch(torch)
        ui.flash.setImageResource(if (torch) R.drawable.ic_flash_on else R.drawable.ic_flash_off)
    }

    // ---------------------------------------------------------------- live mode
    private fun setLive(on: Boolean) {
        ui.overlay.live = on
        tracker.reset()
        ui.liveChips.removeAllViews()
        ui.liveScroll.visibility = if (on) View.VISIBLE else View.GONE
        ui.hint.setText(if (on) R.string.scan_live_hint else R.string.scan_hint)
        if (hasCamera()) bindCamera()
    }

    private fun analyze(image: ImageProxy) {
        val eng = engine
        if (eng == null || busy || !analyzing.compareAndSet(false, true)) { image.close(); return }
        try {
            val bmp = ImageUtils.rotate(image.toBitmap(), image.imageInfo.rotationDegrees)
            image.close()
            val results = eng.read(ImageUtils.toRgb(bmp), prefs.threshold(eng.config.conf))
            _ui?.overlay?.setResults(results, bmp.width, bmp.height)
            val confirmed = tracker.update(results)
            if (confirmed.isNotEmpty()) {
                val crops = confirmed.map { ImageUtils.crop(bmp, it) }
                activity?.runOnUiThread { onLiveConfirmed(confirmed, crops) }
            }
        } catch (e: Exception) {
            image.close()
        } finally {
            analyzing.set(false)
        }
    }

    private fun onLiveConfirmed(plates: List<PlateResult>, crops: List<Bitmap>) {
        val u = _ui ?: return
        plates.forEachIndexed { i, r ->
            val chip = Chip(requireContext()).apply {
                text = PlateText.display(r.text)
                typeface = androidx.core.content.res.ResourcesCompat.getFont(context, R.font.vazirmatn_bold)
                chipIcon = null
                setOnClickListener { showResult(crops[i], listOf(r), 0) }
            }
            u.liveChips.addView(chip, 0)
            if (prefs.autoSave) history.add(r.text, r.box.score, r.textConf, crops[i])
        }
        u.hint.text = getString(R.string.live_found, Fmt.fa(tracker.count))
        buzz()
    }

    private fun buzz() {
        if (!prefs.vibrate) return
        val v = requireContext().getSystemService(Vibrator::class.java)
        if (v?.hasVibrator() == true) v.vibrate(VibrationEffect.createOneShot(35, VibrationEffect.DEFAULT_AMPLITUDE))
        else _ui?.root?.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
    }

    // ---------------------------------------------------------------- photo / gallery
    private fun takePhoto() {
        val ic = capture ?: return openGallery()
        if (busy) return
        setBusy(true)
        ic.takePicture(ContextCompat.getMainExecutor(requireContext()), object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val rot = image.imageInfo.rotationDegrees
                val bmp = image.toBitmap()
                image.close()
                viewLifecycleOwner.lifecycleScope.launch {
                    processPhoto(withContext(Dispatchers.Default) { ImageUtils.limit(ImageUtils.rotate(bmp, rot)) })
                }
            }

            override fun onError(e: ImageCaptureException) {
                setBusy(false)
                ui.hint.text = getString(R.string.error, e.message)
            }
        })
    }

    private fun openGallery() = pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))

    private suspend fun decode(uri: Uri): Bitmap? = withContext(Dispatchers.IO) {
        try {
            val b = if (Build.VERSION.SDK_INT >= 28) {
                ImageDecoder.decodeBitmap(ImageDecoder.createSource(requireContext().contentResolver, uri)) { d, _, _ ->
                    d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                }
            } else {
                @Suppress("DEPRECATION") MediaStore.Images.Media.getBitmap(requireContext().contentResolver, uri)
            }
            ImageUtils.limit(b)
        } catch (e: Exception) {
            null
        }
    }

    private suspend fun processPhoto(bmp: Bitmap) {
        val eng = engine ?: Engines.get(requireContext()).also { engine = it }
        setBusy(true)
        val t0 = System.currentTimeMillis()
        val results = withContext(Dispatchers.Default) { eng.read(ImageUtils.toRgb(bmp), prefs.threshold(eng.config.conf)) }
        val ms = System.currentTimeMillis() - t0
        android.util.Log.i("PlateReader", "photo ${bmp.width}x${bmp.height}: total $ms ms (detect ${eng.lastDetectMs} ms, recognize ${eng.lastRecognizeMs} ms), ${results.size} plates")
        setBusy(false)
        if (results.isNotEmpty()) buzz()
        if (prefs.autoSave) withContext(Dispatchers.IO) { results.forEach { history.add(it.text, it.box.score, it.textConf, ImageUtils.crop(bmp, it)) } }
        showResult(bmp, results, ms)
    }

    private fun showResult(bmp: Bitmap, results: List<PlateResult>, ms: Long) {
        ResultStore.set(bmp, results, ms, ImageUtils.annotate(bmp, results, PlateDrawing(requireContext())))
        if (childFragmentManager.findFragmentByTag("result") == null) ResultSheet().show(childFragmentManager, "result")
    }

    private fun setBusy(b: Boolean) {
        busy = b
        _ui?.let {
            it.busy.visibility = if (b) View.VISIBLE else View.GONE
            it.hint.setText(if (b) R.string.processing else if (it.liveChip.isChecked) R.string.scan_live_hint else R.string.scan_hint)
        }
    }
}
