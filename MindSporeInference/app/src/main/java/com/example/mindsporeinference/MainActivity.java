package com.example.mindsporeinference;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;

import com.huawei.hms.mlsdk.common.MLException;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            ModelRunner modelRunner = new ModelRunner();
            modelRunner.runInference();
        } catch (MLException e) {
            throw new RuntimeException(e);
        }
    }
}