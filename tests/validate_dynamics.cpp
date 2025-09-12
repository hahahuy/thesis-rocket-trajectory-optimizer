#include "physics/dynamics.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace physics;

int main() {
    // Parameters
    AscentDynamics::Params p;
    p.A = 1.5;
    p.Isp = 300.0;
    p.Tmax = 180000.0;
    p.m_dry = 1500.0;
    p.m_prop = 4500.0;
    p.enable_wind = false;
    p.earth.latitude = 28.5 * M_PI / 180.0; // e.g., KSC

    // Optional multi-stage example (2 stages)
    physics::propulsion::MultiStageVehicle vehicle;
    physics::propulsion::Stage st1{1500.0, 4500.0, {180000.0, 300.0, 220000.0, 320.0}, 60.0};
    physics::propulsion::Stage st2{800.0, 2500.0, {90000.0, 315.0, 110000.0, 340.0}, 45.0};
    vehicle.stages = {st1, st2};
    vehicle.current_stage = 0;
    p.vehicle = &vehicle;

    // Initial state
    AscentDynamics::State s0{0.0, 0.0, 0.0, 0.0, p.m_dry + p.m_prop};

    // Control function by stage
    auto control_func = [&](double t, int stage) -> AscentDynamics::Control {
        (void)t; // time not used in this simple demo
        double theta = M_PI / 2.0; // vertical
        double T = p.Tmax;
        if (stage == 1) {
            T = 90000.0; // second stage thrust example
        }
        return {T, theta};
    };

    // Integrate with staging
    auto traj = ForwardIntegrator::integrate_rk45_with_staging(
        AscentDynamics::rhs, s0, control_func, p,
        0.0, 120.0, 1e-6, 1e-8, 0.2
    );

    // Write CSV
    std::ofstream out("trajectory.csv");
    out << std::fixed << std::setprecision(6);
    out << "t,x,y,vx,vy,m,q,mach\n";

    for (auto &ts : traj) {
        double t = ts.first;
        const auto &s = ts.second;
        double v = std::hypot(s.vx, s.vy);
        // quick density and speed of sound estimates for reporting
        double rho = AscentDynamics::atmospheric_density(s.y, p);
        double q = 0.5 * rho * v * v;
        double a = std::sqrt(1.4 * 287.05287 * 288.15);
        double mach = (a > 1e-9) ? v / a : 0.0;
        out << t << "," << s.x << "," << s.y << "," << s.vx << "," << s.vy << "," << s.m << "," << q << "," << mach << "\n";
    }
    out.close();

    std::cout << "Wrote trajectory.csv with " << traj.size() << " samples\n";
    return 0;
}
