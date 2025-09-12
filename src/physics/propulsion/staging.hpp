#pragma once

#include <vector>

namespace physics {
namespace propulsion {

struct EngineParams {
    double sea_level_thrust;   // N
    double sea_level_isp;      // s
    double vacuum_thrust;      // N
    double vacuum_isp;         // s
};

struct Stage {
    double dry_mass;        // kg
    double propellant_mass; // kg
    EngineParams engine;
    double burn_time;       // s
};

struct MultiStageVehicle {
    std::vector<Stage> stages;
    int current_stage = 0;
};

class StagingLogic {
public:
    static bool should_separate(double mass_current, const MultiStageVehicle &vehicle);
    static void perform_separation(double &mass_current, MultiStageVehicle &vehicle);
};

} // namespace propulsion
} // namespace physics


