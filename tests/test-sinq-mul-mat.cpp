#include "../src/llama-model.h"

#include <ggml.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

template <typename T>
float get_value(const ggml_tensor * tensor, int row, int col) {
    const auto * base = static_cast<const char *>(tensor->data);
    return *reinterpret_cast<const T *>(base + row * tensor->nb[1] + col * tensor->nb[0]);
}

template <typename T>
void set_value(ggml_tensor * tensor, int row, int col, float value) {
    auto * base = static_cast<char *>(tensor->data);
    *reinterpret_cast<T *>(base + row * tensor->nb[1] + col * tensor->nb[0]) = value;
}

} // namespace

int main() {
    llama_model_params params = llama_model_default_params();
    llama_model model(params);

    const int nrows = 4;
    const int ncols = 3;
    const int n_tokens = 2;

    const std::vector<float> row_scale = {1.5f, 0.5f, 2.0f, 1.2f};
    const std::vector<float> col_scale = {0.8f, 1.3f, 0.6f};

    std::vector<float> original = {
        1.0f,  2.0f,  3.0f,
        4.0f,  5.0f,  6.0f,
        7.0f,  8.0f,  9.0f,
       10.0f, 11.0f, 12.0f,
    };

    std::vector<float> normalized(nrows * ncols);
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            normalized[r * ncols + c] = original[r * ncols + c] / (row_scale[r] * col_scale[c]);
        }
    }

    llama_model_test_set_sinq_scales(model, "weight", row_scale, col_scale);

    const size_t mem_size = 1u << 18;
    std::vector<uint8_t> buffer(mem_size);
    ggml_init_params init_params = {};
    init_params.mem_size   = buffer.size();
    init_params.mem_buffer = buffer.data();
    ggml_context * ctx = ggml_init(init_params);
    assert(ctx != nullptr);

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_set_name(weight, "weight");
    std::memcpy(weight->data, normalized.data(), normalized.size() * sizeof(float));

    ggml_tensor * weight_original = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_set_name(weight_original, "weight_original");
    std::memcpy(weight_original->data, original.data(), original.size() * sizeof(float));

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, n_tokens);
    ggml_set_name(input, "input");
    for (int t = 0; t < n_tokens; ++t) {
        for (int c = 0; c < ncols; ++c) {
            set_value<float>(input, t, c, static_cast<float>(c * n_tokens + t + 1));
        }
    }

    ggml_tensor * expected = ggml_mul_mat(ctx, weight_original, input);
    ggml_set_name(expected, "expected");

    ggml_tensor * actual = model.mul_mat_with_sinq(ctx, weight, input);
    ggml_set_name(actual, "actual");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, expected);
    ggml_build_forward_expand(gf, actual);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    const float tol = 1e-6f;
    const int dim0 = static_cast<int>(expected->ne[0]);
    const int dim1 = static_cast<int>(expected->ne[1]);
    for (int i1 = 0; i1 < dim1; ++i1) {
        for (int i0 = 0; i0 < dim0; ++i0) {
            float exp_val = get_value<float>(expected, i1, i0);
            float act_val = get_value<float>(actual, i1, i0);
            if (std::fabs(exp_val - act_val) > tol) {
                ggml_free(ctx);
                return 1;
            }
        }
    }

    ggml_free(ctx);
    return 0;
}
