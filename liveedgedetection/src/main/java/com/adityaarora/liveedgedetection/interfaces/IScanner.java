package com.adityaarora.liveedgedetection.interfaces;

import android.graphics.Bitmap;
import android.graphics.Paint;

import com.adityaarora.liveedgedetection.enums.ScanHint;

import org.opencv.core.Point;

/**
 * Interface between activity and surface view
 */

public interface IScanner {
    void setPaintAndBorder(ScanHint scanHint, Paint paint, Paint border);
    void displayHint(ScanHint scanHint);
    void onPictureClicked(Bitmap bitmap, Point[] points);
}
