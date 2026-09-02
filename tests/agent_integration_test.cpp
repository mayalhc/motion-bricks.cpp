#include <motionbricks/motionbricks.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

#ifndef MOTIONBRICKS_REFERENCE_BUNDLE
#error MOTIONBRICKS_REFERENCE_BUNDLE is required
#endif
#ifndef MOTIONBRICKS_REFERENCE_STYLES
#error MOTIONBRICKS_REFERENCE_STYLES is required
#endif

namespace {
bool check(mb_status status, const char * operation, const char * error) {
    if (status == MB_OK) return true;
    std::cerr << operation << ": " << mb_status_string(status) << ": " << error << '\n';
    return false;
}
}

int main(int argc, char ** argv) {
    const bool vulkan = argc == 2 && std::string_view(argv[1]) == "vulkan";
    char error[1024]{};
    mb_runtime_options * options = nullptr;
    mb_model * model = nullptr;
    mb_style * style = nullptr;
    mb_agent * agent = nullptr;
    mb_command * command = nullptr;
    mb_motion * motion = nullptr;
    if (!check(mb_runtime_options_create(&options, error, sizeof error), "options", error) ||
        !check(mb_runtime_options_set_device(options, vulkan ? MB_DEVICE_VULKAN : MB_DEVICE_CPU,
                                             error, sizeof error), "device", error) ||
        !check(mb_model_load(MOTIONBRICKS_REFERENCE_BUNDLE, options, &model, error, sizeof error),
               "model", error) ||
        !check(mb_style_load(model, MOTIONBRICKS_REFERENCE_STYLES "/walk.mbstyle", &style,
                             error, sizeof error), "style", error) ||
        !check(mb_agent_create(model, &agent, error, sizeof error), "agent", error) ||
        !check(mb_agent_reset(agent, style, error, sizeof error), "reset", error) ||
        !check(mb_command_create(&command, error, sizeof error), "command", error) ||
        !check(mb_command_set_style(command, style, error, sizeof error), "set style", error) ||
        !check(mb_command_set_movement_direction(command, 0.0F, 0.0F, 1.0F,
                                                  error, sizeof error), "movement", error) ||
        !check(mb_command_set_facing_direction(command, 0.0F, 0.0F, 1.0F,
                                                error, sizeof error), "facing", error) ||
        !check(mb_command_set_seed(command, 23U, error, sizeof error), "seed", error) ||
        !check(mb_agent_plan(agent, command, &motion, error, sizeof error), "plan", error))
        return 1;
    std::uint64_t frames = 0, joints = 0, root_values = 0, rotation_values = 0;
    std::uint64_t target_frames = 0, target_root_values = 0, target_rotation_values = 0;
    const float * roots = nullptr;
    const float * rotations = nullptr;
    const float * target_roots = nullptr;
    const float * target_rotations = nullptr;
    if (!check(mb_motion_get_frame_count(motion, &frames, error, sizeof error), "frames", error) ||
        !check(mb_motion_get_joint_count(motion, &joints, error, sizeof error), "joints", error) ||
        !check(mb_motion_get_root_translations(motion, &roots, &root_values, error, sizeof error),
               "roots", error) ||
        !check(mb_motion_get_local_rotations_xyzw(motion, &rotations, &rotation_values,
                                                  error, sizeof error), "rotations", error) ||
        !check(mb_motion_get_target_frame_count(motion, &target_frames, error, sizeof error),
               "target frames", error) ||
        !check(mb_motion_get_target_root_translations(motion, &target_roots, &target_root_values,
                                                      error, sizeof error), "target roots", error) ||
        !check(mb_motion_get_target_local_rotations_xyzw(motion, &target_rotations,
                                                         &target_rotation_values,
                                                         error, sizeof error), "target rotations", error))
        return 1;
    bool valid = frames >= 24U && frames <= 64U && frames % 4U == 0U && joints == 34U &&
                 root_values == frames * 3U && rotation_values == frames * joints * 4U &&
                 target_frames == 4U && target_root_values == target_frames * 3U &&
                 target_rotation_values == target_frames * joints * 4U;
    valid = valid && std::all_of(roots, roots + root_values,
        [](float value) { return std::isfinite(value); });
    valid = valid && std::all_of(rotations, rotations + rotation_values,
        [](float value) { return std::isfinite(value); });
    valid = valid && std::all_of(target_roots, target_roots + target_root_values,
        [](float value) { return std::isfinite(value); });
    valid = valid && std::all_of(target_rotations, target_rotations + target_rotation_values,
        [](float value) { return std::isfinite(value); });
    float maximum_norm_error = 0.0F;
    for (std::uint64_t index = 0; index < frames * joints; ++index) {
        const float * q = rotations + index * 4U;
        maximum_norm_error = std::max(maximum_norm_error,
            std::abs(std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]) - 1.0F));
    }
    valid = valid && maximum_norm_error < 2.0e-4F;
    for (std::uint64_t index = 0; index < target_frames * joints; ++index) {
        const float * q = target_rotations + index * 4U;
        maximum_norm_error = std::max(maximum_norm_error,
            std::abs(std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]) - 1.0F));
    }
    valid = valid && maximum_norm_error < 2.0e-4F;
    std::cout << "agent " << (vulkan ? "vulkan" : "cpu") << " frames=" << frames
              << " quaternion_norm_error=" << maximum_norm_error << '\n';
    if (valid) valid = check(mb_agent_advance(agent, 4U, error, sizeof error), "advance", error);
    mb_motion_free(motion);
    mb_command_free(command);
    mb_agent_free(agent);
    mb_style_free(style);
    mb_model_free(model);
    mb_runtime_options_free(options);
    return valid ? 0 : 1;
}
