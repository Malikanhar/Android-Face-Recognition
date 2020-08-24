package com.example.kilam.androidfacerecognition

import android.app.Dialog
import android.content.Context
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.view.WindowManager
import org.opencv.android.*
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import org.opencv.core.Scalar
import com.example.kilam.androidfacerecognition.Custom.CameraBridgeViewBase
import org.opencv.core.Core
import android.content.pm.ActivityInfo
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.view.Window
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import kotlinx.android.synthetic.main.activity_detection.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.collections.ArrayList

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private var TAG = "MainActivityOpenCV"
    private var MODEL_PATH = "MobileFacenet.tflite"

    private var ID_CAMERA = CameraBridgeViewBase.CAMERA_ID_FRONT

    private lateinit var mRGBA : Mat
    private lateinit var javaCameraView : CameraBridgeViewBase
    private lateinit var cascadeClassifier: CascadeClassifier
    private lateinit var mCascadeFile : File
    private lateinit var mBitmap: Bitmap

    private lateinit var tfliteModel : MappedByteBuffer
    private lateinit var interpreter : Interpreter
    private lateinit var tImage : TensorImage
    private lateinit var tBuffer : TensorBuffer

    private lateinit var persons : ArrayList<Person>

    // Width of the image that our model expects
    var inputImageWidth = 112

    // Height of the image that our model expects
    var inputImageHeight = 112

    private val IMAGE_MEAN = 127.5f
    private val IMAGE_STD = 128f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_detection)

        initializeModel()
        initializePerson()

        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV is Configured or Connected Successfully")
        } else {
            Log.d(TAG, "OpenCV is not working or Loaded")
        }

        javaCameraView = findViewById(R.id.opencv_camera)
        javaCameraView.visibility = SurfaceView.VISIBLE
        javaCameraView.setCvCameraViewListener(this@MainActivity)

        btn_register.setOnClickListener { showDialog() }
    }

    private fun showDialog(){
        onPause()

        val dialog = Dialog(this@MainActivity)
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE)
        dialog.setCancelable(false)
        dialog.setContentView(R.layout.dialog_register)

        val iv_person = dialog.findViewById(R.id.iv_capture) as ImageView
        iv_person.setImageBitmap(mBitmap)

        val et_name = dialog.findViewById(R.id.et_name) as EditText
        val btnSave = dialog.findViewById(R.id.btn_save) as Button
        val btnCancel  = dialog.findViewById(R.id.btn_cancel) as Button

        btnSave.setOnClickListener {
            val embedding = getEmbedding(mBitmap)
            val name = et_name.text.toString().trim()
            persons.add(Person(name, embedding))
            dialog.dismiss()
            onResume()
        }

        btnCancel.setOnClickListener {
            dialog.dismiss()
            onResume()
        }

        dialog.show()
    }

    private fun initializePerson(){
        persons = ArrayList()
    }

    private fun initializeModel(){
        try{
            tfliteModel = loadModelFile()

            @Suppress("DEPRECATION")
            interpreter = Interpreter(tfliteModel)

            val probabilityTensorIndex = 0
            val probabilityShape = interpreter.getOutputTensor(probabilityTensorIndex).shape() // {1, EMBEDDING_SIZE}
            val probabilityDataType = interpreter.getOutputTensor(probabilityTensorIndex).dataType()

            // Creates the input tensor
            tImage = TensorImage(DataType.FLOAT32)

            // Creates the output tensor and its processor
            tBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)

            Log.d(TAG, "Model loaded successful")
        } catch (e : IOException){
            Log.e(TAG, "Error reading model", e)
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = assets.openFd(MODEL_PATH)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startOffset: Long = fileDescriptor.startOffset
        val declaredLength: Long = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun recognize(embedding : FloatArray) : String {
        return if (persons.isNotEmpty()){
            val similarities = ArrayList<Float>()
            persons.forEach {
                similarities.add(cosineSimilarity(it.embedding, embedding))
            }
            val maxVal = similarities.max()!!
            if (maxVal > 0.8) "${persons[similarities.indexOf(maxVal)].name} ${(maxVal * 100).toString().take(2)} %"
            else "unknown"
        } else "unknown"
    }

    private fun getEmbedding(bitmap : Bitmap) : FloatArray {
        tImage = loadImage(bitmap)

        interpreter.run(tImage.buffer, tBuffer.buffer.rewind())

        return tBuffer.floatArray
    }

    private fun cosineSimilarity(A: FloatArray?, B: FloatArray?): Float {
        if (A == null || B == null || A.isEmpty() || B.isEmpty() || A.size != B.size) {
            return 2.0F
        }

        var sumProduct = 0.0
        var sumASq = 0.0
        var sumBSq = 0.0
        for (i in A.indices) {
            sumProduct += (A[i] * B[i]).toDouble()
            sumASq += (A[i] * A[i]).toDouble()
            sumBSq += (B[i] * B[i]).toDouble()
        }
        return if (sumASq == 0.0 && sumBSq == 0.0) {
            2.0F
        } else (sumProduct / (Math.sqrt(sumASq) * Math.sqrt(sumBSq))).toFloat()
    }

    private fun loadImage(bitmap : Bitmap) : TensorImage {
        // Loads bitmap into a TensorImage
        tImage.load(bitmap)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputImageHeight, inputImageWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(IMAGE_MEAN, IMAGE_STD))
            .build()
        return imageProcessor.process(tImage)
    }

    private var baseLoaderCallback = object : BaseLoaderCallback(this@MainActivity) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    try {
                        val inputStream = resources.openRawResource(R.raw.haarcascade_frontalface_alt2)
                        val cascadeDir = getDir("cascade", Context.MODE_PRIVATE)
                        mCascadeFile = File(cascadeDir, "haarcascade_frontalface_alt2.xml")
                        val os = FileOutputStream(mCascadeFile)

                        val buffer = ByteArray(4096)
                        var bytesRead : Int?

                        do {
                            bytesRead = inputStream.read(buffer)
                            if (bytesRead == -1) break
                            os.write(buffer, 0, bytesRead)
                        } while (true)

                        inputStream.close()
                        os.close()

                        cascadeClassifier = CascadeClassifier(mCascadeFile.absolutePath)
                        if (cascadeClassifier.empty()) {
                            Log.d(TAG, "Failed to load cascade classifier")
                        } else
                            Log.d(TAG,"Loaded cascade classifier from " + mCascadeFile.absolutePath)

                        cascadeDir.delete()
                    } catch (e: IOException) {
                        e.printStackTrace()
                        Log.e(TAG, "Failed to load cascade. Exception thrown: $e")
                    }

                    javaCameraView.setCameraIndex(ID_CAMERA)
                    javaCameraView.enableView()
                }
                else ->
                    super.onManagerConnected(status)
            }
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        mRGBA = Mat()
    }

    override fun onCameraViewStopped() {
        mRGBA.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        mRGBA = inputFrame.rgba()

        // Flip image to get mirror effect
        val orientation = javaCameraView.screenOrientation
        if (javaCameraView.isEmulator)
        // Treat emulators as a special case
            Core.flip(mRGBA, mRGBA, 1) // Flip along y-axis
        else {
            when (orientation) {
                ActivityInfo.SCREEN_ORIENTATION_PORTRAIT, ActivityInfo.SCREEN_ORIENTATION_REVERSE_PORTRAIT ->{
                    if (ID_CAMERA == CameraBridgeViewBase.CAMERA_ID_BACK) Core.flip(mRGBA, mRGBA,1) // Flip along x-axis 1
                    Core.flip(mRGBA, mRGBA,0) // Flip along x-axis 0
                }
                ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE, ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE ->{
                    if (ID_CAMERA == CameraBridgeViewBase.CAMERA_ID_BACK) Core.flip(mRGBA, mRGBA,0) // Flip along x-axis 0
                    Core.flip(mRGBA, mRGBA,1) // Flip along y-axis 1
                }
            }
        }

        Core.rotate(mRGBA , mRGBA , Core.ROTATE_90_COUNTERCLOCKWISE)

        // Detect faces
        val detectedFaces = MatOfRect()
        cascadeClassifier.detectMultiScale(mRGBA, detectedFaces, 1.1, 2, 2, Size(112.0, 112.0), Size())

        // Draw rectangle for detected faces
        for (rect : Rect in detectedFaces.toArray()) {
            val m = mRGBA.submat(rect)
            mBitmap = Bitmap.createBitmap(m.width(),m.height(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(m, mBitmap)
            val res = recognize(getEmbedding(mBitmap))
            val scalar = if (res == "unknown") { Scalar(255.0, 0.0, 0.0) } else Scalar(0.0, 255.0, 0.0)

            Imgproc.rectangle(mRGBA,
                Point(rect.x.toDouble(), rect.y.toDouble()),
                Point(rect.x.toDouble() + rect.width, rect.y.toDouble() + rect.height),
                scalar, 1)

            Imgproc.putText(mRGBA,
                res,
                Point(rect.x.toDouble(), rect.y.toDouble() - 5.0),
                2,
                1.0,
                scalar, 2)
        }

        Core.rotate(mRGBA , mRGBA , Core.ROTATE_90_CLOCKWISE)

        return mRGBA
    }

    override fun onDestroy() {
        super.onDestroy()
        javaCameraView.disableView()
    }

    override fun onPause() {
        super.onPause()
        javaCameraView.disableView()
    }

    override fun onResume() {
        super.onResume()
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV is Configured or Connected Successfully")
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        } else {
            Log.d(TAG, "OpenCV is not working or Loaded")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, baseLoaderCallback)
        }
    }

    companion object {

        // Used to load the 'native-lib' library on application startup.
        init {
            System.loadLibrary("native-lib")
        }
    }
}

