
#include <vector>

double ApplyCatboostModel(const std::vector<float>& floatFeatures);

extern "C" {
    void predict_batch(const float* x, float* out, int n_rows, int n_cols) {
        std::vector<float> features(n_cols);
        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < n_cols; ++j) {
                features[j] = x[i * n_cols + j];
            }
            out[i] = (float)ApplyCatboostModel(features);
        }
    }
}
