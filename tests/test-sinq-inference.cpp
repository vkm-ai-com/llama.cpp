#include "get-model.h"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

struct generation_result {
    std::vector<llama_token> tokens;
    std::string text;
};

std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & prompt) {
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n_prompt < 0) {
        return {};
    }

    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true) < 0) {
        tokens.clear();
    }
    return tokens;
}

std::string token_to_string(const llama_vocab * vocab, llama_token token) {
    char buffer[256];
    const int written = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, true);
    if (written < 0) {
        return {};
    }
    return std::string(buffer, static_cast<size_t>(written));
}

generation_result generate_from_model(llama_model * model, const std::string & prompt, int n_predict) {
    generation_result result;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (vocab == nullptr) {
        return result;
    }

    std::vector<llama_token> prompt_tokens = tokenize_prompt(vocab, prompt);
    if (prompt_tokens.empty()) {
        return result;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = std::max<int>(prompt_tokens.size() + n_predict + 16, 32);
    ctx_params.n_batch = prompt_tokens.size();

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        return result;
    }

    auto sampler_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    if (sampler == nullptr) {
        llama_free(ctx);
        return result;
    }

    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        llama_sampler_free(sampler);
        llama_free(ctx);
        return result;
    }

    while (static_cast<int>(result.tokens.size()) < n_predict) {
        const llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        result.tokens.push_back(new_token);
        result.text += token_to_string(vocab, new_token);
        llama_sampler_accept(sampler, new_token);

        const llama_batch next = llama_batch_get_one(&result.tokens.back(), 1);
        if (llama_decode(ctx, next) != 0) {
            break;
        }
    }

    llama_sampler_free(sampler);
    llama_free(ctx);

    return result;
}

std::filesystem::path make_temp_path() {
    const auto tmp_dir = std::filesystem::temp_directory_path();
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto unique = std::filesystem::path("llama-sinq-test-" + std::to_string(now) + ".gguf");
    auto candidate = tmp_dir / unique;
    while (std::filesystem::exists(candidate)) {
        const auto alt = std::chrono::steady_clock::now().time_since_epoch().count();
        unique = std::filesystem::path("llama-sinq-test-" + std::to_string(alt) + ".gguf");
        candidate = tmp_dir / unique;
    }
    return candidate;
}

} // namespace

int main(int argc, char ** argv) {
    char * model_path_c = get_model_or_exit(argc, argv);
    const std::string model_path(model_path_c);

    ggml_backend_load_all();
    llama_backend_init();
    struct backend_guard {
        ~backend_guard() {
            llama_backend_free();
        }
    } backend_guard;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    llama_model * baseline_model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (baseline_model == nullptr) {
        std::fprintf(stderr, "failed to load baseline model from %s\n", model_path.c_str());
        return 1;
    }

    const std::string prompt = "how are you?";
    const int n_predict = 16;

    const generation_result baseline = generate_from_model(baseline_model, prompt, n_predict);
    llama_model_free(baseline_model);

    if (baseline.tokens.empty()) {
        std::fprintf(stderr, "baseline generation failed\n");
        return 1;
    }

    const std::filesystem::path tmp_path = make_temp_path();

    llama_model_quantize_params q_params = llama_model_quantize_default_params();
    q_params.nthread = 1;
    q_params.ftype   = LLAMA_FTYPE_ALL_F16;
    q_params.use_sinq = true;

    if (llama_model_quantize(model_path.c_str(), tmp_path.string().c_str(), &q_params) != 0) {
        std::fprintf(stderr, "model quantization failed\n");
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        return 1;
    }

    llama_model * sinq_model = llama_model_load_from_file(tmp_path.string().c_str(), model_params);
    if (sinq_model == nullptr) {
        std::fprintf(stderr, "failed to load quantized model from %s\n", tmp_path.string().c_str());
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        return 1;
    }

    const generation_result sinq_result = generate_from_model(sinq_model, prompt, n_predict);
    llama_model_free(sinq_model);

    std::error_code ec;
    std::filesystem::remove(tmp_path, ec);

    if (sinq_result.tokens != baseline.tokens) {
        std::fprintf(stderr, "generation mismatch\n");
        std::fprintf(stderr, "baseline: %s\n", baseline.text.c_str());
        std::fprintf(stderr, "sinq: %s\n", sinq_result.text.c_str());
        return 1;
    }

    return 0;
}
