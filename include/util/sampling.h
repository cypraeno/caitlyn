#ifndef SAMPLING_H
#define SAMPLING_H

// ATTENTION:
// credit: OSL testrender ©
// NOT OUR CODE, IN RELEASE, THIS MUST BE REPLACED WITH A PROPER CREDIT!!!


struct MIS {
    // for the function below, enumerate the cases for:
    // the sampled function being a weight or eval,
    // the "other" function being a weight or eval
    enum MISMode { WEIGHT_WEIGHT, WEIGHT_EVAL, EVAL_WEIGHT };

    template<MISMode mode> static inline float power_heuristic(float pdf1, float pdf2) {
        float num = pow(pdf1, mode);
        float denom = pow(pdf1, mode) + pow(pdf2, mode);
        return num / denom;
    }
};

#endif