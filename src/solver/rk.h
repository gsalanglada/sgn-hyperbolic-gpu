#ifndef SGN_RK_H
#define SGN_RK_H

// =============================================================================
// rk.h -- explicit Runge-Kutta time integration (CPU reference integrator)
// =============================================================================
//
// WHAT THIS FILE IS
//   The time integrators of the CPU solvers. They advance the semi-discrete
//   system dU/dt = R(t, U) produced by the spatial discretization.
//
//   Dimension-agnostic: they operate on a flat state vector and a user-supplied
//   right-hand-side callback dudt = f(t, u). The SAME integrator serves the 1D
//   and 2D CPU solvers and the MPI CPU solvers -- only the callback changes.
//   The GPU solvers re-implement the identical scheme (same stages, same
//   coefficients, same controller) as device kernels, so that CPU and GPU runs
//   take the same accept/reject decisions (see src/cuda/*.cu).
//
//   Direct port of the NSK-FR-CUDA integrator (the reused numerical core).
//
// SCHEMES
//   rk4Step            classical fixed-step RK4 (reference / convergence tests)
//   bs3Step            one fixed step of the production Bogacki-Shampine pair
//   integrateAdaptive  embedded Bogacki-Shampine 3(2) with FSAL, a PI step-size
//                      controller, a hard dtMin abort, and both L2 and max-norm
//                      error monitoring. This is the production integrator.
// =============================================================================

#include <functional>
#include <vector>

namespace sgn {

// Flat state vector: all degrees of freedom of the problem, variable-major
// (all h values, then all hu values, ...).
using State = std::vector<double>;

// Right-hand-side callback: fill dudt (already sized like u) with R(t, u).
using RhsFn = std::function<void(double t, const State& u, State& dudt)>;

// One classical RK4 step of size dt, in place (u <- u(t + dt)).
void rk4Step(const RhsFn& f, double t, double dt, State& u);

// One fixed-step Bogacki-Shampine step of size dt, in place: the SAME stages
// and the SAME third-order update used by integrateAdaptive, with the embedded
// estimate and the PI controller simply not evaluated. Used by the temporal
// convergence study to measure the order of the production integrator itself.
// (Recomputing k1 each call is numerically identical to the FSAL reuse inside
// integrateAdaptive, since the reused stage is f at exactly the new state.)
void bs3Step(const RhsFn& f, double t, double dt, State& u);

// Tuning knobs of the adaptive integrator.
struct AdaptiveRKOptions {
  double atol = 1e-8;     // absolute tolerance in the error scale atol + rtol*|u|
  double rtol = 1e-6;     // relative tolerance
  double dtMin = 1e-13;   // hard floor: if the controller asks for dt < dtMin, ABORT
  double dtMax = 1e30;    // optional cap on the step size
  double safety = 0.9;    // safety factor S of the step-size update
  double facMin = 0.2;    // never shrink faster than this in one step
  double facMax = 5.0;    // never grow faster than this in one step
};

// What happened during an adaptive integration.
struct AdaptiveRKStats {
  long accepted = 0;        // accepted steps
  long rejected = 0;        // rejected (and retried) steps
  long rhsEvals = 0;        // total right-hand-side evaluations (cost measure)
  double tReached = 0.0;    // time of the last accepted state
  double dtLast = 0.0;      // last accepted step (or the rejected dt on abort)
  double maxNormLast = 0.0; // scaled max-norm error of the last accepted step
  bool aborted = false;     // true if dt fell below dtMin (NOT a clean finish)
};

// Integrate u from t0 to tEnd with an initial step guess dt0 (typically a CFL
// estimate from the solver). On dt < dtMin the integration stops with
// stats.aborted = true and u / tReached left at the last accepted state: the
// caller decides what to do, instead of the run grinding on at a useless step.
AdaptiveRKStats integrateAdaptive(const RhsFn& f, State& u,
                                  double t0, double tEnd, double dt0,
                                  const AdaptiveRKOptions& opt = {});

} // namespace sgn

#endif // SGN_RK_H
