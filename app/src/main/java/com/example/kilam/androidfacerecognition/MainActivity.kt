package com.example.kilam.androidfacerecognition

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

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private var TAG = "MainActivityOpenCV"

    private lateinit var mRGBA : Mat
    private lateinit var mGray : Mat
    private lateinit var mRGBAT : Mat
    private lateinit var javaCameraView : CameraBridgeViewBase
    private lateinit var cascadeClassifier: CascadeClassifier
    private lateinit var mCascadeFile : File

    override fun onCameraViewStarted(width: Int, height: Int) {
        mRGBA = Mat()
        mGray = Mat()
        Log.d(TAG, "height of images : ${height} and width : ${width}")
    }

    override fun onCameraViewStopped() {
        mRGBA.release()
        mGray.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        mRGBA = inputFrame.rgba()

        mRGBAT = mRGBA.t()
        Core.flip(mRGBA.t(), mRGBAT, 1)
        Imgproc.resize(mRGBAT, mRGBAT, mRGBA.size())

        // Detect faces
        val detectedFaces = MatOfRect()
        cascadeClassifier.detectMultiScale(mRGBAT, detectedFaces)

        // Draw rectangle for detected faces
        for (rect : Rect in detectedFaces.toArray()) {
            Imgproc.rectangle(mRGBAT, Point(rect.x.toDouble(), rect.y.toDouble()),
                Point(rect.x.toDouble() + rect.width, rect.y.toDouble() + rect.height),
                Scalar(255.0, 0.0, 0.0)
            )
        }

        return mRGBAT
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_detection)

        window.setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN)

        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV is Configured or Connected Successfully")
        } else {
            Log.d(TAG, "OpenCV is not working or Loaded")
        }

        javaCameraView = findViewById(R.id.opencv_camera)
        javaCameraView.visibility = SurfaceView.VISIBLE
        javaCameraView.setCvCameraViewListener(this@MainActivity)
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

                    javaCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK)
                    javaCameraView.enableView()
                }
                else ->
                    super.onManagerConnected(status)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        if (javaCameraView != null) {
            javaCameraView.disableView()
        }
    }

    override fun onPause() {
        super.onPause()
        if (javaCameraView != null) {
            javaCameraView.disableView()
        }
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

