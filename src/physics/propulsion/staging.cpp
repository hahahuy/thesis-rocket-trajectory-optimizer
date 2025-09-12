#include "physics/propulsion/staging.hpp"

namespace physics {
namespace propulsion {

bool StagingLogic::should_separate(double mass_current, const MultiStageVehicle &vehicle) {
    if (vehicle.current_stage < 0 || vehicle.current_stage >= static_cast<int>(vehicle.stages.size())) return false;
    const Stage &st = vehicle.stages[vehicle.current_stage];
    double stage_dry = st.dry_mass;
    // Separate when mass is within small epsilon of next stage mass stack (heuristic)
    return mass_current <= stage_dry + 1e-3;
}

void StagingLogic::perform_separation(double &mass_current, MultiStageVehicle &vehicle) {
    if (vehicle.current_stage < 0 || vehicle.current_stage >= static_cast<int>(vehicle.stages.size())) return;
    const Stage &st = vehicle.stages[vehicle.current_stage];
    // Drop current stage dry mass
    mass_current -= st.dry_mass;
    vehicle.current_stage += 1;
}

} // namespace propulsion
} // namespace physics


