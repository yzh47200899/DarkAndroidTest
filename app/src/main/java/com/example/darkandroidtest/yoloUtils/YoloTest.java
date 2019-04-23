package com.example.darkandroidtest.yoloUtils;

import android.graphics.Bitmap;

public class YoloTest {
    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("darknetlib");
    }
    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native double testyolo(String imgfile);
    public native double testBitmap(Bitmap dst);
    public native int testImageInfo(Bitmap dst,String imgfile);
}
