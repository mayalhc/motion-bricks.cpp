#include "decoder.hpp"
#include "handles.hpp"
#include "model.hpp"

#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#ifndef MOTIONBRICKS_REFERENCE_BUNDLE
#error MOTIONBRICKS_REFERENCE_BUNDLE is required
#endif
#ifndef MOTIONBRICKS_REFERENCE_FIXTURES
#error MOTIONBRICKS_REFERENCE_FIXTURES is required
#endif

namespace {
struct gguf_deleter { void operator()(gguf_context * value) const { gguf_free(value); } };
struct ggml_deleter { void operator()(ggml_context * value) const { ggml_free(value); } };
template<class T> std::vector<T> tensor_values(ggml_context * context, const char * name) {
    auto * tensor = ggml_get_tensor(context, name);
    assert(tensor != nullptr);
    std::vector<T> values(static_cast<std::size_t>(ggml_nelements(tensor)));
    std::memcpy(values.data(), tensor->data, values.size() * sizeof(T));
    return values;
}
}

int main(int argc, char ** argv) {
    const bool vulkan = argc == 2 && std::string_view(argv[1]) == "vulkan";
    mb_model model;
    mb_runtime_options options;
    options.device = vulkan ? MB_DEVICE_VULKAN : MB_DEVICE_CPU;
    std::string reason;
    auto status = motionbricks::detail::load_model_bundle(
        MOTIONBRICKS_REFERENCE_BUNDLE, &options, model, reason);
    if (status != MB_OK) { std::cerr << reason << '\n'; return 1; }
    ggml_context * raw = nullptr;
    const std::string fixture_path = std::string(MOTIONBRICKS_REFERENCE_FIXTURES) + "/vq-decoder.gguf";
    std::unique_ptr<gguf_context, gguf_deleter> fixture(
        gguf_init_from_file(fixture_path.c_str(), {false, &raw}));
    std::unique_ptr<ggml_context, ggml_deleter> tensors(raw);
    assert(fixture && tensors);
    const auto quantized = tensor_values<float>(tensors.get(), "input.quantized");
    const auto external = tensor_values<float>(tensors.get(), "input.external_cond");
    const auto target = tensor_values<float>(tensors.get(), "input.target_cond");
    const auto mask_i32 = tensor_values<std::int32_t>(tensors.get(), "input.has_target_cond");
    const auto expected = tensor_values<float>(tensors.get(), "output.motion");
    std::vector<std::uint8_t> mask(mask_i32.begin(), mask_i32.end());
    std::vector<float> actual;
    std::vector<motionbricks::detail::decoder_trace> traces;
    status = motionbricks::detail::run_vq_decoder(
        *model.runtime, quantized, external, target, mask, 6, actual, reason, &traces);
    if (status != MB_OK) { std::cerr << reason << '\n'; return 1; }
    assert(actual.size() == expected.size());
    float max_absolute = 0.0F, max_relative = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        max_absolute = std::max(max_absolute, std::abs(actual[index] - expected[index]));
        max_relative = std::max(max_relative,
            std::abs(actual[index] - expected[index]) / std::max(1.0e-6F, std::abs(expected[index])));
    }
    std::cout << "decoder " << (vulkan ? "vulkan" : "cpu")
              << " max_abs=" << max_absolute << " max_rel=" << max_relative << '\n';
    for (const auto & trace : traces) {
        const auto reference = tensor_values<float>(
            tensors.get(), ("trace." + trace.name).c_str());
        float difference = 0.0F;
        for (std::size_t index = 0; index < reference.size(); ++index)
            difference = std::max(difference, std::abs(reference[index] - trace.values[index]));
        std::cout << trace.name << " max_abs=" << difference << '\n';
    }
    return max_absolute <= (vulkan ? 1.0e-4F : 3.0e-4F) ? 0 : 1;
}
