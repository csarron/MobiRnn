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
    float* valB = (float*) rsGetElementAt(b_in, y);
    sum += (*valB);
    return fmax(sum, 0.0f);// relu
}

uint32_t hiddenUnites;
uint32_t outputDims;

rs_allocation w_out;
rs_allocation b_out;

rs_allocation label_prob;

rs_allocation matA; // shape: m * sameDim
rs_allocation matB; // shape: sameDim * n
rs_allocation matAB; // shape: m * n
uint32_t sameDim;

//two matrix multiplication, can still work if one of them is vector
void matMul(float *v_out, uint32_t x, uint32_t y) {
  float sum = 0.0f;
  //rsDebug("sameDim=", sameDim);
  for (uint32_t c = 0; c < sameDim; c++) {
  // caution: x=m, y=n, x means number of columns, y means number of rows
    float* a = (float*) rsGetElementAt(matA, c, y);
    float* b = (float*) rsGetElementAt(matB, x, c);
    //rsDebug("a=", (*a));
    //rsDebug("b=", (*b));
    //rsDebug("c=", c);
    //rsDebug("mul=",  (*a) * (*b));
    sum += (*a) * (*b);
    //rsDebug("sum=", sum);
  }
  rsSetElementAt(matAB, &sum, x, y);
}

// two matrix addition, or two vector addition
void matAdd(float *v_out, uint32_t x, uint32_t y) {
    float sum = 0.0f;
    float* valA = (float*) rsGetElementAt(matA, x, y);
    float* valB = (float*) rsGetElementAt(matB, x, y);
    sum = (*valA) * (*valB);
    rsSetElementAt(matAB, &sum, x, y);
}

float RS_KERNEL caclC(float a, float b){
    return a + b;
}

float RS_KERNEL caclH(float a, float b){
    return a + b;
}

static inline float sigmoid(float x){
    return 1 / (1 + exp(-x));
}
//void compute(){
// 	int start = rsUptimeNanos();
//
//	int inputX = rsAllocationGetDimX(inputRaw);
//	int inputY = rsAllocationGetDimY(inputRaw);
//	int wInX = rsAllocationGetDimX(w_in);
//	int wInY = rsAllocationGetDimY(w_in);
//	rsAllocationCopy2DRange(matA, 0, 0, 0, NULL, inputX, inputY, inputRaw, 0, 0, 0, NULL);
//	rsAllocationCopy2DRange(matB, 0, 0, 0, NULL, wInX, wInY, w_in,  0, 0, 0, NULL);
//
//	matAB = rsCreateAllocation_float(inputX, wInY);
//	rsForEach(matMul, matA, matB);
//	int end = rsUptimeNanos();
// 	int cost = end - start;
// 	rsDebug("time cost(ns)=", cost);
//
// 	for(int i=0; i < inputX; i++){
// 	    for(int j=0; j < wInY; j++){
// 	        rsDebug("matAB=", *((float*) rsGetElementAt(matAB, i, j)));
// 	    }
// 	}
//}