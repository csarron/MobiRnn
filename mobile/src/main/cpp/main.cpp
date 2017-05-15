#include <android/log.h>
#include <Eigen/Dense>
#include <jni.h>

#ifndef LOG_FLAG
#define LOG_FLAG true
#endif // #ifndef LOG_FLAG

void empty(...) {
}

#define LOG(...) \
  (LOG_FLAG ? ((void)__android_log_print(ANDROID_LOG_INFO, "native::", __VA_ARGS__)): ((void) empty(__VA_ARGS__)))

using namespace Eigen;

float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

ArrayXXf sigmoid(ArrayXXf arr) {
    arr *= -1;
    return (arr.exp() + 1).inverse();
}

int argMax(float *x, int len) {
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

extern "C"
JNIEXPORT jint JNICALL
Java_com_cscao_apps_mobirnn_model_CpuModel_predictNativeEigen(JNIEnv *env, jobject instance,
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

    const int layer_size = config[0];
    const int time_step = config[1];
    const int hidden_unit = config[2];
    const int input_dim = config[3];
    const int output_dim = config[4];

    LOG("layer_size: %d", layer_size);
    LOG("time_step: %d", time_step);
    LOG("hidden_unit: %d", hidden_unit);
    LOG("input_dim: %d", input_dim);
    LOG("output_dim: %d", output_dim);

    Map<MatrixXf> inputMat(input, input_dim, time_step);
    Map<MatrixXf> winMat(wIn, hidden_unit, input_dim);
    Map<VectorXf> binMat(bIn, hidden_unit);
    Map<MatrixXf> woutMat(wOut, output_dim, hidden_unit);
    Map<VectorXf> boutMat(bOut, output_dim);
    Map<MatrixXf> weightsMat(weights, (4 * hidden_unit) * (2 * hidden_unit), layer_size);
    Map<MatrixXf> biasesMat(biases, 4 * hidden_unit, layer_size);

    VectorXf c = VectorXf::Zero(hidden_unit);
    VectorXf h = VectorXf::Zero(hidden_unit);
    // input activation
    MatrixXf inputs = MatrixXf::Zero(hidden_unit, time_step);
    inputs = ((winMat * inputMat).colwise() + binMat).array().max(0).matrix();

    MatrixXf concat = MatrixXf::Zero(hidden_unit, 2);
//    MatrixXf linearResults = MatrixXf::Zero(hidden_unit, 4);

    // cell computation
    for (int layer = 0; layer < layer_size; ++layer) {
        c.setZero();
        h.setZero();
        Map<MatrixXf> layerWeights(weightsMat.col(layer).data(), (4 * hidden_unit),
                                   (2 * hidden_unit));
        for (int step = 0; step < time_step; ++step) {
            concat.col(0) = inputs.col(step);
            concat.col(1) = h;
            VectorXf linearResults = layerWeights * Map<VectorXf>(concat.data(), concat.size()) +
                                     biasesMat.col(layer);

            Map<MatrixXf> linearMap(linearResults.data(), hidden_unit, 4);
            Map<VectorXf> i(linearMap.col(0).data(), hidden_unit);
            Map<VectorXf> j(linearMap.col(1).data(), hidden_unit);
            Map<VectorXf> f(linearMap.col(2).data(), hidden_unit);
            Map<VectorXf> o(linearMap.col(3).data(), hidden_unit);
            c = (c.array() * sigmoid(f.array() + 1) +
                 sigmoid(i.array()) * (j.array().tanh())).matrix();
            h = (c.array().tanh() * sigmoid(o.array())).matrix();
            inputs.col(step) = h;

        }
    }

    //output activation
    VectorXf outProb =  woutMat * h + boutMat;

    int label;
    outProb.maxCoeff(&label);

    env->ReleaseFloatArrayElements(input_, input, 0);
    env->ReleaseIntArrayElements(config_, config, 0);
    env->ReleaseFloatArrayElements(wIn_, wIn, 0);
    env->ReleaseFloatArrayElements(bIn_, bIn, 0);
    env->ReleaseFloatArrayElements(wOut_, wOut, 0);
    env->ReleaseFloatArrayElements(bOut_, bOut, 0);
    env->ReleaseFloatArrayElements(weights_, weights, 0);
    env->ReleaseFloatArrayElements(biases_, biases, 0);
    return (label + 1);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_cscao_apps_mobirnn_model_CpuModel_predictNative(JNIEnv *env, jobject instance,
                                                         jfloatArray input_, jint layer_size,
                                                         jint time_steps, jint hidden_unites,
                                                         jint in_dim, jint out_dim,
                                                         jfloatArray wIn_,
                                                         jfloatArray bIn_,
                                                         jfloatArray convertedWOut_,
                                                         jfloatArray bOut_,
                                                         jfloatArray convertedWeights_,
                                                         jfloatArray convertedBiases_) {


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

    return argMax(label_prob, out_dim) + 1;
}