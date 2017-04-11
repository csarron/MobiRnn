#pragma version(1)
#pragma rs java_package_name(com.cscao.apps.mobirnn)

rs_allocation inputRaw;
uint32_t timeSteps;
uint32_t inputDims;

rs_allocation w_in;
rs_allocation b_in;

float RS_KERNEL input_transform(uint32_t x, uint32_t y){
    float sum = 0.0f;
    for (uint32_t c = 0; c < inputDims; c++) {
        float* a = (float*) rsGetElementAt(inputRaw, c, y);
        float* b = (float*) rsGetElementAt(w_in, x, c);
        sum += (*a) * (*b);
    }
    float* valB = (float*) rsGetElementAt(b_in, x);
    sum += (*valB);
    return fmax(sum, 0.0f);// relu
}

rs_allocation weights;
rs_allocation biases;
uint32_t layerSize;
uint32_t hiddenUnites;
rs_allocation inputs;  // timeSteps * hiddenUnits
//rs_allocation outputs;

float RS_KERNEL set_zeros(uint32_t x, uint32_t y){
    return 0.0f;
}

static void print(rs_allocation mat){
    const uint32_t dimX =  rsAllocationGetDimX(mat);
    const uint32_t dimY =  rsAllocationGetDimY(mat);
    rsDebug("dimX: ", dimX);
    rsDebug("dimY: ", dimY);
    for (uint32_t x = 0; x < dimX; x++){
        for (uint32_t y = 0; y < dimY; y++){
            float* a = (float*) rsGetElementAt(mat, x, y);
            rsDebug("x: ", x);
            rsDebug("y: ", y);
            rsDebug("mat: ", (*a));
         }     
    }
}

rs_allocation c;
rs_allocation h;

rs_allocation input_concat;
rs_allocation layer_weights;
rs_allocation layer_biases;
rs_allocation linear_result;
uint32_t current_layer;
uint32_t current_step;

float RS_KERNEL concat_in(uint32_t x){
    return *((float*) rsGetElementAt(inputs, x, current_step));
}

void RS_KERNEL concat_h(float in, uint32_t x){
    rsSetElementAt_float(input_concat, in, hiddenUnites + x);
}

float RS_KERNEL linear_map(uint32_t x){
    float sum = 0.0f;
    //rsDebug("current_layer=",  current_layer);
    for (uint32_t c = 0; c < hiddenUnites * 2; c++) {
        float* a = (float*) rsGetElementAt(input_concat, c, 0); // input_concat is dimY is 1
        float* b = (float*) rsGetElementAt(weights, x, c, current_layer);
        //rsDebug("a=",  (*a));
        //rsDebug("b=",  (*b));
        sum += (*a) * (*b);
        //rsDebug("sum=",  sum);
    }

    float* valB = (float*) rsGetElementAt(biases, x, current_layer);
    sum += (*valB);
    return sum;
}

rs_allocation i_gate;
rs_allocation j_val; // C_hat in original formula
rs_allocation f_gate;
rs_allocation o_gate;

float RS_KERNEL get_i(uint32_t x){
    return *((float*) rsGetElementAt(linear_result, x, 0));
}

float RS_KERNEL get_j(uint32_t x){
    return *((float*) rsGetElementAt(linear_result, hiddenUnites + x, 0));
}

float RS_KERNEL get_f(uint32_t x){
    return *((float*) rsGetElementAt(linear_result, hiddenUnites * 2 + x, 0));
}

float RS_KERNEL get_o(uint32_t x){
    return *((float*) rsGetElementAt(linear_result, hiddenUnites * 3 + x, 0));
}

static inline float sigmoid(float x){
     return 1.0f / (1.0f + exp(-x));
}

float RS_KERNEL pointwise_c(uint32_t x){
    float cVal = *((float*) rsGetElementAt(c, x));
    float fVal = *((float*) rsGetElementAt(f_gate, x));
    float iVal = *((float*) rsGetElementAt(i_gate, x));
    float jVal = *((float*) rsGetElementAt(j_val, x));

    float newC = cVal * sigmoid(fVal + 1) + sigmoid(iVal) * tanh(jVal);
    //rsDebug("newC:", newC);
    return newC;
}

float RS_KERNEL pointwise_h(uint32_t x){
    float cVal = *((float*) rsGetElementAt(c, x));
    float oVal = *((float*) rsGetElementAt(o_gate, x));

    return tanh(cVal) * sigmoid(oVal);
}

void RS_KERNEL update_input(float in, uint32_t x){
    rsSetElementAt_float(inputs, in, x, current_step);
}

// float RS_KERNEL set_output(float in, uint32_t x){
//     return in;
// }

rs_allocation w_out;
rs_allocation b_out;

rs_allocation label_prob;

float RS_KERNEL output_transform(uint32_t x){
    float sum = 0.0f;
    for (uint32_t c = 0; c < hiddenUnites; c++) {
        float* a = (float*) rsGetElementAt(h, c);
        float* b = (float*) rsGetElementAt(w_out, x, c);
        sum += (*a) * (*b);
    }
    float* valB = (float*) rsGetElementAt(b_out, x);
    sum += (*valB);
    return sum;
}

void predict(){
    rsForEach(input_transform, inputs);
    //print(inputs);

    c = rsCreateAllocation_float(hiddenUnites);
    h = rsCreateAllocation_float(hiddenUnites);

    input_concat = rsCreateAllocation_float(hiddenUnites * 2, 1);
    layer_weights = rsCreateAllocation_float(hiddenUnites * 4, hiddenUnites * 2);
    layer_biases = rsCreateAllocation_float(hiddenUnites * 4, 1);
    linear_result = rsCreateAllocation_float(hiddenUnites * 4, 1);

    i_gate = rsCreateAllocation_float(hiddenUnites);
    j_val = rsCreateAllocation_float(hiddenUnites);
    f_gate = rsCreateAllocation_float(hiddenUnites);
    o_gate = rsCreateAllocation_float(hiddenUnites);

    //outputs = rsCreateAllocation_float(hiddenUnites);
    //rsForEach(set_zeros, outputs);
    for (uint32_t layer = 0; layer < layerSize; layer++) {//
        rsForEach(set_zeros, c);
        rsForEach(set_zeros, h);
        current_layer = layer;

        for (uint32_t t = 0; t < timeSteps; t++) {//
            // concat in_ and h_ to input_concat
            current_step = t;
            rsForEach(concat_in, input_concat); // set in to first part of input_concat
            rsForEach(concat_h, h); // set h to last part of input_concat
            //print(input_concat);

            rsForEach(linear_map, linear_result);
            //print(linear_result);

            // split linear_result to get i,j,f,o
            rsForEach(get_i, i_gate);
            rsForEach(get_j, j_val);
            rsForEach(get_f, f_gate);
            rsForEach(get_o, o_gate);
            // for (uint32_t k = 0; k < hiddenUnites; k++) {
            //     rsDebug("i_gate=", *((float*) rsGetElementAt(i_gate, k)));
            //     rsDebug("j_val=", *((float*) rsGetElementAt(j_val, k)));
            //     rsDebug("f_gate=", *((float*) rsGetElementAt(f_gate, k)));
            //     rsDebug("o_gate=", *((float*) rsGetElementAt(o_gate, k)));
            // }
            rsForEach(pointwise_c, c);
            rsForEach(pointwise_h, h);

            // update inputs
            rsForEach(update_input, h);
            //rsForEach(set_output, h, outputs);//not need to set outputs, h is output

        }
    }
    // for (uint32_t k = 0; k < hiddenUnites; k++) {
    //     // rsDebug("c=", *((float*) rsGetElementAt(c, k)));
    //     rsDebug("h=", *((float*) rsGetElementAt(h, k)));
    //     // rsDebug("outputs=", *((float*) rsGetElementAt(outputs, k)));
    // }
    rsForEach(output_transform, label_prob);
}


// rs_allocation matA; // shape: m * sameDim
// rs_allocation matB; // shape: sameDim * n
// rs_allocation matAB; // shape: m * n
// uint32_t sameDim;

// //two matrix multiplication, can still work if one of them is vector
// void matMul(float *v_out, uint32_t x, uint32_t y) {
//   float sum = 0.0f;
//   //rsDebug("sameDim=", sameDim);
//   for (uint32_t c = 0; c < sameDim; c++) {
//   // caution: x=m, y=n, x means number of columns, y means number of rows
//     float* a = (float*) rsGetElementAt(matA, c, y);
//     float* b = (float*) rsGetElementAt(matB, x, c);
//     //rsDebug("a=", (*a));
//     //rsDebug("b=", (*b));
//     //rsDebug("c=", c);
//     //rsDebug("mul=",  (*a) * (*b));
//     sum += (*a) * (*b);
//     //rsDebug("sum=", sum);
//   }
//   rsSetElementAt_float(matAB, sum, x, y);
// }

// // two matrix addition, or two vector addition
// void matAdd(float *v_out, uint32_t x, uint32_t y) {
//     float sum = 0.0f;
//     float* valA = (float*) rsGetElementAt(matA, x, y);
//     float* valB = (float*) rsGetElementAt(matB, x, y);
//     sum = (*valA) * (*valB);
//     rsSetElementAt_float(matAB, sum, x, y);
// }
