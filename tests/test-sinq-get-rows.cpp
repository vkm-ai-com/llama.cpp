#include "../src/llama-model.h"

#include <ggml.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

int main() {
    llama_model_params params = llama_model_default_params();
    llama_model model(params);

    const int nrows = 4;
    const int ncols = 3;

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

    const int n_ids = 3;
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_ids);
    ggml_set_name(ids, "ids");
    auto * ids_data = static_cast<int32_t *>(ids->data);
    ids_data[0] = 2;
    ids_data[1] = 0;
    ids_data[2] = 3;
    ggml_set_input(ids);

    ggml_tensor * result = model.get_rows_with_sinq(ctx, weight, ids);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    auto get_value = [&](int token, int col) {
        return *reinterpret_cast<float *>(
            static_cast<char *>(result->data) + token * result->nb[1] + col * result->nb[0]);
    };

    const float tol = 1e-6f;
    for (int t = 0; t < n_ids; ++t) {
        int row_index = ids_data[t];
        for (int c = 0; c < ncols; ++c) {
            float expected = original[row_index * ncols + c];
            float actual   = get_value(t, c);
            if (std::fabs(expected - actual) > tol) {
                ggml_free(ctx);
                return 1;
            }
        }
    }

    ggml_free(ctx);
    return 0;
}
