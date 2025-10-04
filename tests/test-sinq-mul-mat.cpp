#include "../src/llama-model.h"

#include <ggml.h>

#include "../ggml/src/ggml-quants.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

int main() {
    llama_model_params params = llama_model_default_params();
    llama_model model(params);

    const int nrows = 3;
    const int ncols = 32; // multiple of Q4_0 block size
    const int n_tokens = 2;

    const std::vector<float> row_scale = {1.2f, 0.7f, 1.5f};
    const std::vector<float> col_scale = {
        0.8f, 1.1f, 0.9f, 1.3f,
        1.05f, 0.95f, 1.2f, 0.85f,
        1.15f, 0.75f, 1.4f, 0.65f,
        0.9f, 1.25f, 1.05f, 0.7f,
        1.1f, 0.88f, 1.3f, 0.92f,
        1.18f, 0.97f, 0.83f, 1.22f,
        0.78f, 1.07f, 0.99f, 1.26f,
        1.11f, 0.93f, 1.04f, 0.87f,
    };

    std::vector<float> original(nrows * ncols);
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            original[r * ncols + c] = 0.1f * (1 + r) * (1 + c);
        }
    }

    std::vector<float> normalized(nrows * ncols);
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            normalized[r * ncols + c] = original[r * ncols + c] / (row_scale[r] * col_scale[c]);
        }
    }

    llama_model_test_set_sinq_scales(model, "weight", row_scale, col_scale);

    const size_t mem_size = 1u << 18;
    std::vector<uint8_t> buffer(mem_size);
    ggml_init_params init_params{};
    init_params.mem_size = buffer.size();
    init_params.mem_buffer = buffer.data();
    ggml_context * ctx = ggml_init(init_params);
    assert(ctx != nullptr);

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, ncols, nrows);
    ggml_set_name(weight, "weight");

    const int blocks_per_row = ncols / ggml_blck_size(GGML_TYPE_Q4_0);
    std::vector<block_q4_0> quantized((size_t) nrows * blocks_per_row);
    for (int r = 0; r < nrows; ++r) {
        quantize_row_q4_0_ref(
            normalized.data() + (size_t) r * ncols,
            quantized.data()   + (size_t) r * blocks_per_row,
            ncols);
    }
    std::memcpy(weight->data, quantized.data(), quantized.size() * sizeof(block_q4_0));

    std::vector<float> input_data(ncols * n_tokens);
    for (int c = 0; c < ncols; ++c) {
        for (int t = 0; t < n_tokens; ++t) {
            input_data[c * n_tokens + t] = 0.05f * (1 + c + t);
        }
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, n_tokens);
    std::memcpy(input->data, input_data.data(), input_data.size() * sizeof(float));
    ggml_set_input(input);

    ggml_tensor * result = model.mul_mat_with_sinq(ctx, weight, input);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    const size_t deq_size = (size_t) nrows * ncols;
    std::vector<float> dequantized(deq_size);
    for (int r = 0; r < nrows; ++r) {
        dequantize_row_q4_0(
            reinterpret_cast<const block_q4_0 *>(quantized.data() + (size_t) r * blocks_per_row),
            dequantized.data() + (size_t) r * ncols,
            ncols);
    }

    auto expected = [&](int row, int token) {
        float sum = 0.0f;
        for (int c = 0; c < ncols; ++c) {
            const float scaled_input = input_data[c * n_tokens + token] * col_scale[c];
            sum += dequantized[(size_t) row * ncols + c] * scaled_input;
        }
        sum *= row_scale[row];
        return sum;
    };

    auto actual = [&](int row, int token) {
        return *reinterpret_cast<float *>(
            static_cast<char *>(result->data) + token * result->nb[1] + row * result->nb[0]);
    };

    const float tol = 1e-4f;
    for (int t = 0; t < n_tokens; ++t) {
        for (int r = 0; r < nrows; ++r) {
            if (std::fabs(expected(r, t) - actual(r, t)) > tol) {
                ggml_free(ctx);
                return 1;
            }
        }
    }

    ggml_free(ctx);
    return 0;
}
