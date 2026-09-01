#include "handles.hpp"
#include "model.hpp"
#include "root.hpp"

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
template<class T> std::vector<T> values(ggml_context * context, const char * name) {
    auto * tensor = ggml_get_tensor(context, name); assert(tensor != nullptr);
    std::vector<T> output(static_cast<std::size_t>(ggml_nelements(tensor)));
    std::memcpy(output.data(), tensor->data, output.size() * sizeof(T));
    return output;
}
float maximum_error(const std::vector<float> & actual, const std::vector<float> & expected) {
    assert(actual.size() <= expected.size());
    float result = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index)
        result = std::max(result, std::abs(actual[index] - expected[index]));
    return result;
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
    const std::string path = std::string(MOTIONBRICKS_REFERENCE_FIXTURES) + "/root.gguf";
    std::unique_ptr<gguf_context, gguf_deleter> fixture(gguf_init_from_file(path.c_str(), {false, &raw}));
    std::unique_ptr<ggml_context, ggml_deleter> tensors(raw);
    assert(fixture && tensors);
    const auto global = values<float>(tensors.get(), "input.global_root_values");
    const auto global_mask_i32 = values<std::int32_t>(tensors.get(), "input.has_global_root_values");
    const auto local = values<float>(tensors.get(), "input.local_root_values");
    const auto local_mask_i32 = values<std::int32_t>(tensors.get(), "input.has_local_root_values");
    const auto poses = values<float>(tensors.get(), "input.poses");
    const auto pose_mask_i32 = values<std::int32_t>(tensors.get(), "input.has_poses");
    const auto duration = values<std::int32_t>(tensors.get(), "input.num_tokens");
    const auto expected_logits = values<float>(tensors.get(), "output.num_token_logits");
    const auto expected_root = values<float>(tensors.get(), "output.pred_global_root_values");
    std::vector<std::uint8_t> global_mask(global_mask_i32.begin(), global_mask_i32.end());
    std::vector<std::uint8_t> local_mask(local_mask_i32.begin(), local_mask_i32.end());
    std::vector<std::uint8_t> pose_mask(pose_mask_i32.begin(), pose_mask_i32.end());
    motionbricks::detail::root_result actual;
    status = motionbricks::detail::run_root_planner(
        *model.runtime, global, global_mask, local, local_mask, poses, pose_mask,
        static_cast<std::uint32_t>(duration[0]), actual, reason);
    if (status != MB_OK) { std::cerr << reason << '\n'; return 1; }
    const float logits_error = maximum_error(actual.duration_logits, expected_logits);
    const float root_error = maximum_error(actual.global_root_values, expected_root);
    const auto actual_duration = std::max_element(actual.duration_logits.begin(), actual.duration_logits.end()) -
                                 actual.duration_logits.begin();
    const auto expected_duration = std::max_element(expected_logits.begin(), expected_logits.end()) -
                                   expected_logits.begin();
    std::cout << "root " << (vulkan ? "vulkan" : "cpu")
              << " logits_max_abs=" << logits_error << " motion_max_abs=" << root_error
              << " duration_match=" << (actual_duration == expected_duration) << '\n';
    return std::max(logits_error, root_error) <= (vulkan ? 6.0e-3F : 4.0e-4F) &&
           actual_duration == expected_duration ? 0 : 1;
}
