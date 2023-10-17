package com.example.mindsporeinference;

import android.util.Log;

import com.huawei.hmf.tasks.OnFailureListener;
import com.huawei.hmf.tasks.OnSuccessListener;
import com.huawei.hms.mlsdk.common.MLException;
import com.huawei.hms.mlsdk.custom.MLCustomLocalModel;
import com.huawei.hms.mlsdk.custom.MLModelExecutor;
import com.huawei.hms.mlsdk.custom.MLModelExecutorSettings;
import com.huawei.hms.mlsdk.custom.MLModelInputs;
import com.huawei.hms.mlsdk.custom.MLModelInputOutputSettings;
import com.huawei.hms.mlsdk.custom.MLModelDataType;
import com.huawei.hms.mlsdk.custom.MLModelOutputs;


public class ModelRunner {
    private static final String MODEL_NAME = "simple_nnc_iris";
    private static final String MODEL_FULL_NAME = MODEL_NAME + ".ms";

    private static final String TAG ="ModelRunner";

    private static final int OUTPUT_SIZE = 3;

    MLModelExecutorSettings settings = null;

    MLModelExecutor modelExecutor = null;

    public ModelRunner() throws MLException {
        MLCustomLocalModel localModel = new MLCustomLocalModel.Factory(MODEL_NAME)
                .setAssetPathFile(MODEL_FULL_NAME)
                .create();
        settings = new MLModelExecutorSettings.Factory(localModel).create();
        modelExecutor = MLModelExecutor.getInstance(settings);
    }

    public void runInference() throws MLException {
        // 准备输入数据。
        final float[][] input1 = {
                {7.9f, 3.8f, 6.4f, 2.0f},
        };
        final float[][] input2 = {
                {5.1f, 3.5f, 1.4f, 0.2f},
        };
        MLModelInputs inputs = null;
        try {
            inputs = new MLModelInputs.Factory().add(input1).add(input2).create();
            // 若模型需要多路输入，您需要多次调用add()以便数据能够一次输入到推理器。
        } catch (MLException e) {
            // 处理输入数据格式化异常。
        }

        MLModelInputOutputSettings inOutSettings = new MLModelInputOutputSettings.Factory()
                .setInputFormat(0, MLModelDataType.FLOAT32, new int[]{1, 4})
                // OUTPUT_SIZE: number of categories supported by your model.
                .setOutputFormat(0, MLModelDataType.FLOAT32, new int[]{1, OUTPUT_SIZE})
                .create();

        modelExecutor.exec(inputs, inOutSettings).addOnSuccessListener(new OnSuccessListener<MLModelOutputs>() {
            @Override
            public void onSuccess(MLModelOutputs mlModelOutputs) {
                Log.i(TAG, "OK in inference, continue...");
                float[][] output = mlModelOutputs.getOutput(0);
                // 这里推理的返回结果在output数组里，可以进一步处理。
                Log.i(TAG, "output, " + output[0][0]);
            }
        }).addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(Exception e) {
                // 推理异常。
                Log.e(TAG, "ERROR in inference, exit...");
            }
        });
    }
}
