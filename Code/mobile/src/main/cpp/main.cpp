#include <jni.h>
#include <string>
#include <Eigen/Dense>
#include <math.h>

#define LOG(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, "mobirnn::", __VA_ARGS__))
float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

int argmax(float* x, int len) {
    int max = 0;
    float a = 0.0f;
    for (int i = 0; i < len; i++) {
        if (x[i] > a) {
            a = x[i];
            max = i;
        }
    }
    return max;
}

JNIEXPORT jint JNICALL
Java_com_cscao_apps_mobirnn_model_CpuModel_predictNative(JNIEnv *env, jobject instance,
                                                         jfloatArray input_, jintArray config_,
                                                         jfloatArray wIn_, jfloatArray bIn_,
                                                         jfloatArray wOut_, jfloatArray bOut_,
                                                         jfloatArray weights_,
                                                         jfloatArray biases_) {
    jfloat *input = env->GetFloatArrayElements(input_, NULL);
    jint *config = env->GetIntArrayElements(config_, NULL);
    jfloat *wIn = env->GetFloatArrayElements(wIn_, NULL);
    jfloat *bIn = env->GetFloatArrayElements(bIn_, NULL);
    jfloat *wOut = env->GetFloatArrayElements(wOut_, NULL);
    jfloat *bOut = env->GetFloatArrayElements(bOut_, NULL);
    jfloat *weights = env->GetFloatArrayElements(weights_, NULL);
    jfloat *biases = env->GetFloatArrayElements(biases_, NULL);

    // TODO

    env->ReleaseFloatArrayElements(input_, input, 0);
    env->ReleaseIntArrayElements(config_, config, 0);
    env->ReleaseFloatArrayElements(wIn_, wIn, 0);
    env->ReleaseFloatArrayElements(bIn_, bIn, 0);
    env->ReleaseFloatArrayElements(wOut_, wOut, 0);
    env->ReleaseFloatArrayElements(bOut_, bOut, 0);
    env->ReleaseFloatArrayElements(weights_, weights, 0);
    env->ReleaseFloatArrayElements(biases_, biases, 0);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_cscao_apps_mobirnn_model_Model_predictNative(JNIEnv *env, jobject instance,
                                                      jint layer_size,
                                                      jint time_steps, jint hidden_unites,
                                                      jint in_dim,
                                                      jint out_dim, jfloatArray wIn_,
                                                      jfloatArray bIn_, jfloatArray convertedWOut_,
                                                      jfloatArray bOut_,
                                                      jfloatArray convertedWeights_,
                                                      jfloatArray convertedBiases_,
                                                      jfloatArray input_) {


    jfloat *w_in = env->GetFloatArrayElements(wIn_, NULL);
    jfloat *b_in = env->GetFloatArrayElements(bIn_, NULL);
    jfloat *w_out = env->GetFloatArrayElements(convertedWOut_, NULL);
    jfloat *b_out = env->GetFloatArrayElements(bOut_, NULL);
    jfloat *weights = env->GetFloatArrayElements(convertedWeights_, NULL);
    jfloat *biases = env->GetFloatArrayElements(convertedBiases_, NULL);
    jfloat *inputRaw = env->GetFloatArrayElements(input_, NULL);

    float *inputs = (float *) malloc(time_steps * hidden_unites * sizeof(float));
    memset(inputs, 0, time_steps * hidden_unites * sizeof(float));

    float *c = (float *) malloc(hidden_unites * sizeof(float));
    memset(c, 0, hidden_unites * sizeof(float));

    float *h = (float *) malloc(hidden_unites * sizeof(float));
    memset(h, 0, hidden_unites * sizeof(float));

    float *input_concat = (float *) malloc(2 * hidden_unites * sizeof(float));
    memset(input_concat, 0, 2 * hidden_unites * sizeof(float));

    float *linear_result = (float *) malloc(4 * hidden_unites * sizeof(float));
    memset(linear_result, 0, 4 * hidden_unites * sizeof(float));

    float *label_prob = (float *) malloc(out_dim * sizeof(float));
    memset(label_prob, 0, out_dim * sizeof(float));

    for (int x = 0; x < hidden_unites; x++) {
        for (int y = 0; y < time_steps; y++) {
            float sum = 0.0f;
            for (int dim = 0; dim < in_dim; dim++) {
                float a = *(inputRaw + y * in_dim + dim);
                float b = *(w_in + dim * hidden_unites + x);
                sum += a * b;
            }
            float valB = b_in[x];
            *(inputs + y * hidden_unites + x) = fmaxf(sum + valB, 0.0f);

//            LOG("inputs: %f", *(inputs + y * hidden_unites + x));
        }
    }

    int current_layer;
    int current_step;

    for (int layer = 0; layer < layer_size; layer++) {
        for (int x = 0; x < hidden_unites; x++) {
            h[x] = 0;
            c[x] = 0;
        }

        current_layer = layer;
        for (int t = 0; t < time_steps; t++) {
            current_step = t;
            for (int x = 0; x < hidden_unites; x++) {
                float inVal = *(inputs + current_step * hidden_unites + x);
                input_concat[x] = inVal;
                input_concat[hidden_unites + x] = h[x];
            }

            for (int x = 0; x < hidden_unites * 4; x++) {
                float sum = 0.0f;
                for (int unit = 0; unit < hidden_unites * 2; unit++) {
                    float a = input_concat[unit];
                    float b = *(weights + current_layer * hidden_unites * 4 * hidden_unites * 2 +
                                unit * hidden_unites * 4 + x);
                    sum += a * b;
                }
                float valB = *(biases + current_layer * hidden_unites * 4 + x);
                linear_result[x] = sum + valB;
            }

            for (int x = 0; x < hidden_unites; x++) {
                float cVal = c[x];
                float fVal = linear_result[hidden_unites * 2 + x];
                float iVal = linear_result[x];
                float jVal = linear_result[hidden_unites + x];
                float oVal = linear_result[hidden_unites * 3 + x];
                c[x] = cVal * sigmoid(fVal + 1) + sigmoid(iVal) * tanhf(jVal);
                h[x] = tanhf(c[x]) * sigmoid(oVal);
            }

            for (int x = 0; x < hidden_unites; x++) {
                *(inputs + current_step * hidden_unites + x) = h[x];
            }
        }
    }

    for (int x = 0; x < out_dim; x++) {
        float sum = 0.0f;
        for (int unit = 0; unit < hidden_unites; unit++) {
            float a = h[unit];
            float b = *(w_out + unit * out_dim + x);
            sum += a * b;
        }
        float valB = b_out[x];
        label_prob[x] = sum + valB;
    }

    free(inputs);
    free(input_concat);
    free(c);
    free(h);
    free(linear_result);
    free(label_prob);

    env->ReleaseFloatArrayElements(wIn_, w_in, 0);
    env->ReleaseFloatArrayElements(bIn_, b_in, 0);
    env->ReleaseFloatArrayElements(convertedWOut_, w_out, 0);
    env->ReleaseFloatArrayElements(bOut_, b_out, 0);
    env->ReleaseFloatArrayElements(convertedWeights_, weights, 0);
    env->ReleaseFloatArrayElements(convertedBiases_, biases, 0);
    env->ReleaseFloatArrayElements(input_, inputRaw, 0);

    return argmax(label_prob, out_dim) + 1;
}