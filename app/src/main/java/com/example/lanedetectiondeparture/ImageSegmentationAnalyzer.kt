package com.example.lanedetectiondeparture

import android.content.Context
import android.graphics.*
import android.util.Log
import android.content.Intent
import android.content.pm.PackageManager
import android.os.CountDownTimer
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.lanedetectiondeparture.ml.GreenLaneModel
import io.reactivex.BackpressureStrategy
import io.reactivex.Flowable
import io.reactivex.subjects.PublishSubject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.*
import com.example.lanedetectiondeparture.MainActivity


class ImageSegmentationAnalyzer (): ImageAnalysis.Analyzer {

    companion object {
        private const val DIM_IMG_SIZE_Y: Int = 160
        private const val DIM_IMG_SIZE_X: Int = 80
    }

    data class SegmentationResults(val bitmapMask: Bitmap?, val laneDeparture: String)

    private val random = Random(System.currentTimeMillis())
    private val resultNotifier = PublishSubject.create<SegmentationResults>()


    fun resultsObserver(): Flowable<SegmentationResults> = resultNotifier.toFlowable(BackpressureStrategy.LATEST)

    override fun analyze(image: ImageProxy?, rotationDegrees: Int) {
        try {
            image?.let {
                val ogWidth = image.width
                val ogHeight = image.height

                // 1- prepare the input Bitmap
                val imgJpg = NV21toJPEG(
                    ImageUtils.YUV420toNV21(image.image),
                    image.width,
                    image.height
                )
                val inputBitmap = tfResizeBilinear(
                    BitmapFactory.decodeByteArray(imgJpg, 0, imgJpg.size),
                    DIM_IMG_SIZE_X,
                    DIM_IMG_SIZE_Y,
                    rotationDegrees
                )

                if (inputBitmap == null) {
                    Log.e(TAG, "Input bitmap is null")
                    return
                }

                val w = inputBitmap.width
                val h = inputBitmap.height
                if (w > DIM_IMG_SIZE_X || h > DIM_IMG_SIZE_Y) {
                    Log.e(TAG, String.format("invalid bitmap size: %d x %d [should be: %d x %d]", w, h, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y))
                }

                // 2- load the model
                val model = GreenLaneModel.newInstance(MainApplication.applicationContext())

                // 3- Creates inputs for reference.
                val inputFeature0 = TensorBuffer.createFixedSize(
                    intArrayOf(1, 80, 160, 3),
                    DataType.FLOAT32
                )

                val tensorImage = TensorImage(DataType.FLOAT32)
                tensorImage.load(inputBitmap)
                val byteBuffer = tensorImage.buffer
                inputFeature0.loadBuffer(byteBuffer)
                inputBitmap.recycle()


                // 4- Runs model inference and gets result.
                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer // this a tensorBuffer of shape [1,80,160,1]
                val outputByffer = outputFeature0.buffer


                //5- create and prepare output Bitmap from byteBuffer
                val maskBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                outputByffer.rewind()  // THIS IS SUPER IMPORTANT
                var t = 0
                var Ml = 0
                var Mr = 0
                var control = 0
                val pixs = IntArray(w*h)
                for (i in 0 until w * h) {
                    val g: Float = outputByffer.float!!
                    if ( g >= 0.7f){
                        control++
                        val arrXY = XY(i)
                        if (arrXY[1] == 120){ // we choose y = 120
                            if (t == 0){
                                Ml = arrXY[0]
                                t++
                            }
                            Mr = arrXY[0]
                        }
                        pixs[i] = Color.rgb( 0, 255, 0)
                    }
                    else
                        pixs[i] = Color.TRANSPARENT
                }

                maskBitmap.setPixels(pixs, 0, w, 0, 0, w, h)

                var chaineDeparture = ""
                if(control > 0){
                    val Kl = calculateDist(Ml,120,40,159)
                    val Kr = calculateDist(Mr,120,40,159)
                    chaineDeparture = detectDeparture(Kr, Kl)
                }
                else
                    chaineDeparture = "No Lane Detected"

                // Releases model resources if no longer used.
                model.close()
                // finally, notify results
                resultNotifier.onNext(SegmentationResults(tfResizeBilinear(maskBitmap, ogHeight, ogWidth, 0),chaineDeparture))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Analyzer Error", e)
        }
    }

    fun calculateDist(x1: Int, y1 : Int, x2 : Int, y2 : Int) : Double{
        val distance : Double
        distance = Math.sqrt(Math.pow((x2-x1).toDouble(), 2.0) + Math.pow((y2-y1).toDouble(), 2.0))
        return  distance
    }

    private fun XY (i : Int) : IntArray {
        val arr = IntArray(2)
        arr[1] = (i/80).toInt()
        arr[0] = (i%80) //* 80
        return arr
    }
    fun detectDeparture (dist1 : Double, dist2: Double) : String{ // dist1 : Kr, dist2 : Kl
        var str = when{
            dist1 < dist2 -> "Right Departure"
            dist1 > dist2 -> "Left Departure"
            dist1 == dist2 -> "Normal Driving"
            else ->  "No Lane Detected"
        }
        return str
    }

    private fun tfResizeBilinear(bitmap: Bitmap?, w: Int, h: Int, rotationDegrees: Int): Bitmap? {
        if (bitmap == null) {
            return null
        }
        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        val resized = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        bitmap.recycle()
        val canvas = Canvas(resized)
        canvas.drawBitmap(
            rotated,
            Rect(0, 0, rotated.width, rotated.height),
            Rect(0, 0, w, h),
            null
        )
        rotated.recycle()
        return resized
    }


    fun printt(buf : ByteBuffer){
        buf.rewind()
        for (i in 0 until 80 * 160) {
            val f = buf.getFloat()
            Log.i("TAG", String.format("this the content of buffer %f", f ))
        }
    }

    private fun NV21toJPEG(nv21: ByteArray, width: Int, height: Int): ByteArray {
        val out = ByteArrayOutputStream()
        val yuv = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        yuv.compressToJpeg(Rect(0, 0, width, height), 100, out)
        return out.toByteArray()
    }
}
