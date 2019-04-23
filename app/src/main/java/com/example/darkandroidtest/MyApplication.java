package com.example.darkandroidtest;

import android.app.Application;

import com.example.darkandroidtest.yoloUtils.YoloTest;

public class MyApplication extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        YoloTest yt = new YoloTest();
    }
}
