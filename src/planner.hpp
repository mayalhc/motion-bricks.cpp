#pragma once

#include "motion_rep.hpp"

#include <motionbricks/motionbricks.h>

#include <array>
#include <cstdint>
#include <string>

struct mb_model;
struct mb_motion;

namespace motionbricks::detail {

struct transition_constraints {
    std::array<float, 8U * global_root_width> global_root{};
    std::array<float, 8U * local_root_width> local_root{};
    std::array<float, 8U * external_pose_width> poses{};
    std::array<std::uint8_t, 8U> has_global_root{1,1,1,1,1,1,1,1};
    std::array<std::uint8_t, 8U> has_local_root{1,1,1,0,1,1,1,1};
    std::array<std::uint8_t, 8U> has_poses{1,1,1,1,1,1,1,1};
    std::array<std::uint8_t, 11U> allowed_tokens{1,1,1,1,1,1,1,1,1,1,1};
};

mb_status run_transition(const mb_model & model,
                         const transition_constraints & constraints,
                         mb_motion & output,
                         std::uint32_t * selected_tokens,
                         std::string & reason);

} // namespace motionbricks::detail
