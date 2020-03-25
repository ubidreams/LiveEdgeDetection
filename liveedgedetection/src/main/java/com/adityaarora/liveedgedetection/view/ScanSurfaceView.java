package com.adityaarora.liveedgedetection.view;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.drawable.shapes.PathShape;
import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.FrameLayout;

import com.adityaarora.liveedgedetection.enums.ScanHint;
import com.adityaarora.liveedgedetection.interfaces.IScanner;
import com.adityaarora.liveedgedetection.util.ImageDetectionProperties;
import com.adityaarora.liveedgedetection.util.ScanUtils;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.CvType.CV_8UC1;

/**
 * This class previews the live images from the camera
 */

public class ScanSurfaceView extends FrameLayout implements SurfaceHolder.Callback {
    private static final String TAG = ScanSurfaceView.class.getSimpleName();
    SurfaceView mSurfaceView;
    private final ScanCanvasView scanCanvasView;
    private int vWidth = 0;
    private int vHeight = 0;

    private final Context context;
    private Camera camera;

    private final IScanner iScanner;
    private Camera.Size previewSize;

    private ArrayList<Point[]> bestPoints = new ArrayList<>();

    private boolean isFindingBestCapture = false;

    public ScanSurfaceView(Context context, IScanner iScanner) {
        super(context);
        mSurfaceView = new SurfaceView(context);
        addView(mSurfaceView);
        this.context = context;
        this.scanCanvasView = new ScanCanvasView(context);
        addView(scanCanvasView);
        SurfaceHolder surfaceHolder = mSurfaceView.getHolder();
        surfaceHolder.addCallback(this);
        this.iScanner = iScanner;
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        try {
            requestLayout();
            openCamera();
            this.camera.setPreviewDisplay(holder);
        } catch (IOException e) {
            Log.e(TAG, e.getMessage(), e);
        }
    }

    public void clearAndInvalidateCanvas() {
        scanCanvasView.clear();
        invalidateCanvas();
    }

    public void invalidateCanvas() {
        scanCanvasView.invalidate();
    }

    private void openCamera() {
        if (camera == null) {
            Camera.CameraInfo info = new Camera.CameraInfo();
            int defaultCameraId = 0;
            for (int i = 0; i < Camera.getNumberOfCameras(); i++) {
                Camera.getCameraInfo(i, info);
                if (info.facing == Camera.CameraInfo.CAMERA_FACING_BACK) {
                    defaultCameraId = i;
                }
            }
            camera = Camera.open(defaultCameraId);
            Camera.Parameters cameraParams = camera.getParameters();

            List<String> flashModes = cameraParams.getSupportedFlashModes();
            if (null != flashModes && flashModes.contains(Camera.Parameters.FLASH_MODE_AUTO)) {
                cameraParams.setFlashMode(Camera.Parameters.FLASH_MODE_AUTO);
            }

            camera.setParameters(cameraParams);
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        if (vWidth == vHeight) {
            return;
        }
        if (previewSize == null)
            previewSize = ScanUtils.getOptimalPreviewSize(camera, vWidth, vHeight);

        Camera.Parameters parameters = camera.getParameters();
        camera.setDisplayOrientation(ScanUtils.configureCameraAngle((Activity) context));
        parameters.setPreviewSize(previewSize.width, previewSize.height);

//        if (parameters.getSupportedFocusModes() != null
//                && parameters.getSupportedFocusModes().contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE)) {
//            parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE);
//        } else if (parameters.getSupportedFocusModes() != null
//                && parameters.getSupportedFocusModes().contains(Camera.Parameters.FOCUS_MODE_AUTO)) {
//            parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
//        }

        Camera.Size size = ScanUtils.determinePictureSize(camera, parameters.getPreviewSize());
        parameters.setPictureSize(size.width, size.height);
        parameters.setPictureFormat(ImageFormat.JPEG);

        camera.setParameters(parameters);
        requestLayout();
        setPreviewCallback();
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        stopPreviewAndFreeCamera();
    }

    private void stopPreviewAndFreeCamera() {
        if (camera != null) {
            // Call stopPreview() to stop updating the preview surface.
            camera.stopPreview();
            camera.setPreviewCallback(null);
            // Important: Call release() to release the camera for use by other
            // applications. Applications should release the camera immediately
            // during onPause() and re-open() it during onResume()).
            camera.release();
            camera = null;
        }
    }

    public void setPreviewCallback() {
        this.cancelAutoCapture();
        this.camera.startPreview();
        this.camera.setPreviewCallback(previewCallback);
    }

    private final Camera.PreviewCallback previewCallback = new Camera.PreviewCallback() {
        @Override
        public void onPreviewFrame(byte[] data, Camera camera) {
            if (null != camera && !isFindingBestCapture) {
                try {
                    Camera.Size pictureSize = camera.getParameters().getPreviewSize();
                    Log.d(TAG, "onPreviewFrame - received image " + pictureSize.width + "x" + pictureSize.height);

                    Mat yuv = new Mat(new Size(pictureSize.width, pictureSize.height * 1.5), CV_8UC1);
                    yuv.put(0, 0, data);

                    Mat mat = new Mat(new Size(pictureSize.width, pictureSize.height), CvType.CV_8UC4);
                    Imgproc.cvtColor(yuv, mat, Imgproc.COLOR_YUV2BGR_NV21, 4);
                    yuv.release();

                    Size originalPreviewSize = mat.size();
                    int originalPreviewArea = mat.rows() * mat.cols();

                    Quadrilateral largestQuad = ScanUtils.detectLargestQuadrilateral(mat);
                    clearAndInvalidateCanvas();

                    mat.release();

                    if (null != largestQuad) {
                        drawLargestRect(largestQuad.contour, largestQuad.points, originalPreviewSize, originalPreviewArea);
                    } else {
                        showFindingReceiptHint();
                    }
                } catch (Exception e) {
                    showFindingReceiptHint();
                }
            }
        }
    };

    private void drawLargestRect(MatOfPoint2f approx, final Point[] points, Size stdSize, int previewArea) {
        Path path = new Path();
        // ATTENTION: axis are swapped
        final float previewWidth = (float) stdSize.height;
        final float previewHeight = (float) stdSize.width;

        Log.i(TAG, "previewWidth: " + String.valueOf(previewWidth));
        Log.i(TAG, "previewHeight: " + String.valueOf(previewHeight));

        //Points are drawn in anticlockwise direction
        path.moveTo(previewWidth - (float) points[0].y, (float) points[0].x);
        path.lineTo(previewWidth - (float) points[1].y, (float) points[1].x);
        path.lineTo(previewWidth - (float) points[2].y, (float) points[2].x);
        path.lineTo(previewWidth - (float) points[3].y, (float) points[3].x);
        path.close();

        double area = Math.abs(Imgproc.contourArea(approx));
        Log.i(TAG, "Contour Area: " + String.valueOf(area));

        PathShape newBox = new PathShape(path, previewWidth, previewHeight);
        Paint paint = new Paint();
        Paint border = new Paint();

        //Height calculated on Y axis
        double resultHeight = points[1].x - points[0].x;
        double bottomHeight = points[2].x - points[3].x;
        if (bottomHeight > resultHeight)
            resultHeight = bottomHeight;

        //Width calculated on X axis
        double resultWidth = points[3].y - points[0].y;
        double bottomWidth = points[2].y - points[1].y;
        if (bottomWidth > resultWidth)
            resultWidth = bottomWidth;

        Log.i(TAG, "resultWidth: " + String.valueOf(resultWidth));
        Log.i(TAG, "resultHeight: " + String.valueOf(resultHeight));

        ImageDetectionProperties imgDetectionPropsObj
                = new ImageDetectionProperties(previewWidth, previewHeight, resultWidth, resultHeight,
                previewArea, area, points[0], points[1], points[2], points[3]);

        final ScanHint scanHint;

        if (imgDetectionPropsObj.isDetectedAreaBeyondLimits()) {
            scanHint = ScanHint.FIND_RECT;
            cancelAutoCapture();
        } else if (imgDetectionPropsObj.isDetectedAreaBelowLimits()) {
            cancelAutoCapture();
            if (imgDetectionPropsObj.isEdgeTouching()) {
                scanHint = ScanHint.MOVE_AWAY;
            } else {
                scanHint = ScanHint.MOVE_CLOSER;
            }
        } else if (imgDetectionPropsObj.isDetectedHeightAboveLimit()) {
            cancelAutoCapture();
            scanHint = ScanHint.MOVE_AWAY;
        } else if (imgDetectionPropsObj.isDetectedWidthAboveLimit() || imgDetectionPropsObj.isDetectedAreaAboveLimit()) {
            cancelAutoCapture();
            scanHint = ScanHint.MOVE_AWAY;
        } else {
            if (imgDetectionPropsObj.isEdgeTouching()) {
                cancelAutoCapture();
                scanHint = ScanHint.MOVE_AWAY;
            } else if (imgDetectionPropsObj.isAngleNotCorrect(approx)) {
                cancelAutoCapture();
                scanHint = ScanHint.ADJUST_ANGLE;
            } else {
                Log.i(TAG, "GREEN" + "(resultWidth/resultHeight) > 4: " + (resultWidth / resultHeight) +
                        " points[0].x == 0 && points[3].x == 0: " + points[0].x + ": " + points[3].x +
                        " points[2].x == previewHeight && points[1].x == previewHeight: " + points[2].x + ": " + points[1].x +
                        "previewHeight: " + previewHeight);
                scanHint = ScanHint.CAPTURING_IMAGE;

                camera.autoFocus(new Camera.AutoFocusCallback() {
                    @Override
                    public void onAutoFocus(boolean success, Camera camera) {
                        camera.takePicture(null, null, new Camera.PictureCallback() {
                            public void onPictureTaken(final byte[] data, final Camera camera) {
                                float ratioX = (float)camera.getParameters().getPictureSize().height / previewWidth;
                                float ratioY = (float)camera.getParameters().getPictureSize().width / previewHeight;

                                Point[] pointsArray = new Point[4];
                                pointsArray[0] = new Point((previewWidth - (float) points[3].y) * ratioX, (float) points[3].x * ratioY);
                                pointsArray[1] = new Point((previewWidth - (float) points[0].y) * ratioX, (float) points[0].x * ratioY);
                                pointsArray[2] = new Point((previewWidth - (float) points[1].y) * ratioX, (float) points[1].x * ratioY);
                                pointsArray[3] = new Point((previewWidth - (float) points[2].y) * ratioX, (float) points[2].x * ratioY);

                                saveCaptureAndPoints(data, pointsArray);
                            }
                        });
                    }
                });
            }
        }
        Log.i(TAG, "Preview Area 95%: " + 0.95 * previewArea +
                " Preview Area 20%: " + 0.20 * previewArea +
                " Area: " + String.valueOf(area) +
                " Label: " + scanHint.toString());

        iScanner.displayHint(scanHint);
        iScanner.setPaintAndBorder(scanHint, paint, border);
        scanCanvasView.clear();
        scanCanvasView.addShape(newBox, paint, border);
        invalidateCanvas();
    }

    private void cancelAutoCapture() {
        isFindingBestCapture = false;
        bestPoints.clear();
    }

    private void saveCaptureAndPoints(byte[] data, Point[] points) {
        if (!isFindingBestCapture) {
            saveByteArrayAsFile(data, String.valueOf(bestPoints.size()));
            bestPoints.add(points);

            if (bestPoints.size() == 3) {
                isFindingBestCapture = true;
                autoCapture();
            }
        }
    }

    private void autoCapture() {
        camera.stopPreview();
        camera.setPreviewCallback(null);

        findBestCapture();

        iScanner.displayHint(ScanHint.NO_MESSAGE);
        clearAndInvalidateCanvas();
    }

    private void findBestCapture() {
        int bestIndex = 0;
        double blurLevel = 0;

        for (int i=0; i<bestPoints.size(); i++) {
            File file = new File(context.getCacheDir(), String.valueOf(i));
            Mat mat = Imgcodecs.imread(file.getPath(), CvType.CV_8UC1);

            double nextBlurLevel = blurLevel(mat);

            if (nextBlurLevel > blurLevel) {
                blurLevel = nextBlurLevel;
                bestIndex = i;
            }
        }

        if (blurLevel > 0) {
            final Point[] points = bestPoints.get(bestIndex);
            File bestFile = new File(context.getCacheDir(), String.valueOf(bestIndex));

            Bitmap bitmap = BitmapFactory.decodeFile(bestFile.getPath());;
            Matrix matrix = new Matrix();
            matrix.postRotate(90);
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

            iScanner.onPictureClicked(bitmap, points);
            
            bitmap.recycle();
            for (int j=0; j<bestPoints.size(); j++) {
                new File(context.getCacheDir(), String.valueOf(j)).delete();
            }
        }
    }

    private double blurLevel(Mat image) {
        Mat destination = new Mat();

        Imgproc.Laplacian(image, destination, 3);
        MatOfDouble median = new MatOfDouble();
        MatOfDouble std = new MatOfDouble();
        Core.meanStdDev(destination, median, std);

        return Math.pow(std.get(0, 0)[0], 2.0);
    }

    private void showFindingReceiptHint() {
        iScanner.displayHint(ScanHint.FIND_RECT);
        clearAndInvalidateCanvas();
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        // We purposely disregard child measurements because act as a
        // wrapper to a SurfaceView that centers the camera preview instead
        // of stretching it.
        vWidth = resolveSize(getSuggestedMinimumWidth(), widthMeasureSpec);
        vHeight = resolveSize(getSuggestedMinimumHeight(), heightMeasureSpec);
        setMeasuredDimension(vWidth, vHeight);
        previewSize = ScanUtils.getOptimalPreviewSize(camera, vWidth, vHeight);
    }

    @SuppressWarnings("SuspiciousNameCombination")
    @Override
    protected void onLayout(boolean changed, int l, int t, int r, int b) {
        if (getChildCount() > 0) {

            int width = r - l;
            int height = b - t;

            int previewWidth = width;
            int previewHeight = height;

            if (previewSize != null) {
                previewWidth = previewSize.width;
                previewHeight = previewSize.height;

                int displayOrientation = ScanUtils.configureCameraAngle((Activity) context);
                if (displayOrientation == 90 || displayOrientation == 270) {
                    previewWidth = previewSize.height;
                    previewHeight = previewSize.width;
                }

                Log.d(TAG, "previewWidth:" + previewWidth + " previewHeight:" + previewHeight);
            }

            int nW;
            int nH;
            int top;
            int left;

            float scale = 1.0f;

            // Center the child SurfaceView within the parent.
            if (width * previewHeight < height * previewWidth) {
                Log.d(TAG, "center horizontally");
                int scaledChildWidth = (int) ((previewWidth * height / previewHeight) * scale);
                nW = (width + scaledChildWidth) / 2;
                nH = (int) (height * scale);
                top = 0;
                left = (width - scaledChildWidth) / 2;
            } else {
                Log.d(TAG, "center vertically");
                int scaledChildHeight = (int) ((previewHeight * width / previewWidth) * scale);
                nW = (int) (width * scale);
                nH = (height + scaledChildHeight) / 2;
                top = (height - scaledChildHeight) / 2;
                left = 0;
            }
            mSurfaceView.layout(left, top, nW, nH);
            scanCanvasView.layout(left, top, nW, nH);

            Log.d("layout", "left:" + left);
            Log.d("layout", "top:" + top);
            Log.d("layout", "right:" + nW);
            Log.d("layout", "bottom:" + nH);
        }
    }

    private void saveByteArrayAsFile(byte[] data, String fileName) {
        File file = new File(context.getCacheDir(), fileName);

        try {
            FileOutputStream fos = new FileOutputStream(file);
            fos.write(data);
            fos.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
