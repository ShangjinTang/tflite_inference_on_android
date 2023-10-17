package com.example.tfliteinference;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.File;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        runIrisInference();
    }

    public static int argmax(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private void runIrisInference() {
        try (Interpreter interpreter = new Interpreter(new File("/odm/etc/model_iris.tflite"))) {
            float[][] inputArray = {
                    {7.9f, 3.8f, 6.4f, 2.0f},
                    {5.1f, 3.5f, 1.4f, 0.2f}
            };
            float[][] outputArray = new float[2][3];
            interpreter.run(inputArray, outputArray);

            for (int i = 0; i < 2; i++) {
                float[] probabilities = outputArray[i];
                int predictedClass = argmax(probabilities);
                Log.i("MainActivity", "Prediction for " + i + " is " + predictedClass);
            }
        }
    }
}