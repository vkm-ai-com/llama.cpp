#include "../src/llama-model.h"

#include <ggml.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

int main() {
    llama_model_params params = llama_model_default_params();
    llama_model model(params);

    const int nrows = 16;
    const int ncols = 32;
    const int ntokens = 8;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> original(nrows * ncols);
    for (auto & v : original) {
        v = dist(rng);
    }

    std::vector<float> row_scale(nrows);
    std::vector<float> col_scale(ncols);
    for (auto & v : row_scale) {
        v = 0.5f + std::fabs(dist(rng));
    }
    for (auto & v : col_scale) {
        v = 0.5f + std::fabs(dist(rng));
    }

    std::vector<float> normalized(original.size());
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            normalized[r * ncols + c] = original[r * ncols + c] / (row_scale[r] * col_scale[c]);
        }
    }

    llama_model_test_set_sinq_scales(model, "weight", row_scale, col_scale);

    const size_t qsize = ggml_row_size(GGML_TYPE_Q4_0, ncols) * nrows;
    std::vector<uint8_t> qdata(qsize);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, normalized.data(), qdata.data(), 0, nrows, ncols, nullptr);

    std::vector<float> input_values(ncols * ntokens);
    for (auto & v : input_values) {
        v = dist(rng);
    }

    auto run_matmul = [&](bool use_view) {
        const size_t mem_size = 1u << 22;
        std::vector<uint8_t> buffer(mem_size);
        ggml_init_params init_params = {};
        init_params.mem_size   = buffer.size();
        init_params.mem_buffer = buffer.data();
        ggml_context * ctx = ggml_init(init_params);
        assert(ctx != nullptr);

        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, ncols, nrows);
        ggml_set_name(weight, "weight");
        std::memcpy(weight->data, qdata.data(), qdata.size());

        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, ntokens);
        std::memcpy(input->data, input_values.data(), input_values.size() * sizeof(float));

        ggml_tensor * weight_arg = weight;
        if (use_view) {
            weight_arg = ggml_view_tensor(ctx, weight);
        }

        ggml_tensor * result = model.mul_mat_with_sinq(ctx, weight_arg, input);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, result);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        std::vector<float> output(nrows * ntokens);
        for (int token = 0; token < ntokens; ++token) {
            for (int row = 0; row < nrows; ++row) {
                output[token * nrows + row] = *reinterpret_cast<float *>(
                    static_cast<char *>(result->data) + token * result->nb[1] + row * result->nb[0]);
            }
        }

        ggml_free(ctx);
        return output;
    };

    const std::vector<float> direct = run_matmul(false);
    const std::vector<float> via_view = run_matmul(true);

    bool ok = true;
    const float tol = 5e-6f;
    for (size_t i = 0; i < direct.size(); ++i) {
        if (std::fabs(direct[i] - via_view[i]) > tol * std::max(1.0f, std::fabs(direct[i]))) {
            ok = false;
            break;
        }
    }

    return ok ? 0 : 1;
}
