// =============================================================================
// rk.cpp -- implementation of the CPU Runge-Kutta integrators (see rk.h)
// =============================================================================
//
// Notation used below:
//   u      state at the start of the step (time t)
//   f      right-hand side, f(t, u) = dU/dt
//   k1..k4 stage derivatives
//   tmp    scratch state at which a stage is evaluated
//   dt     step size
//
// Direct port of the NSK-FR-CUDA integrator (the reused numerical core).
// =============================================================================

#include "rk.h"

#include <algorithm>
#include <cmath>

namespace sgn {

// Classical fourth-order Runge-Kutta, four right-hand-side evaluations.
void rk4Step(const RhsFn& f, double t, double dt, State& u) {
  const size_t n = u.size();
  State k1(n), k2(n), k3(n), k4(n), tmp(n);

  f(t, u, k1);                                                      // slope at t
  for (size_t i = 0; i < n; ++i) tmp[i] = u[i] + 0.5 * dt * k1[i];
  f(t + 0.5 * dt, tmp, k2);                                         // midpoint, using k1
  for (size_t i = 0; i < n; ++i) tmp[i] = u[i] + 0.5 * dt * k2[i];
  f(t + 0.5 * dt, tmp, k3);                                         // midpoint, using k2
  for (size_t i = 0; i < n; ++i) tmp[i] = u[i] + dt * k3[i];
  f(t + dt, tmp, k4);                                               // end point, using k3

  // weighted average of the four slopes
  for (size_t i = 0; i < n; ++i)
    u[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

// One fixed step of the Bogacki-Shampine third-order method (three evaluations).
void bs3Step(const RhsFn& f, double t, double dt, State& u) {
  const size_t n = u.size();
  State k1(n), k2(n), k3(n), tmp(n);

  f(t, u, k1);                                                      // slope at t
  for (size_t i = 0; i < n; ++i) tmp[i] = u[i] + 0.5 * dt * k1[i];
  f(t + 0.5 * dt, tmp, k2);                                         // at t + dt/2
  for (size_t i = 0; i < n; ++i) tmp[i] = u[i] + 0.75 * dt * k2[i];
  f(t + 0.75 * dt, tmp, k3);                                        // at t + 3dt/4

  // Third-order update, identical to the accepted solution in integrateAdaptive.
  for (size_t i = 0; i < n; ++i)
    u[i] += dt * (2.0 / 9.0 * k1[i] + 1.0 / 3.0 * k2[i] + 4.0 / 9.0 * k3[i]);
}

// Production adaptive integrator.
//
// Each attempted step:
//   1. stages k2, k3 from k1 (k1 is carried over from the previous step: FSAL)
//   2. third-order candidate uNew
//   3. k4 = f(t+dt, uNew): the "first same as last" stage, needed both for the
//      embedded solution and, if the step is accepted, as the next k1
//   4. embedded second-order solution uEmb and the scaled error of uNew - uEmb
//   5. accept if the L2 error norm <= 1, otherwise reject and retry
//   6. PI controller picks the next dt (after both accepts and rejects)
// Cost: 3 new right-hand-side evaluations per attempted step.
AdaptiveRKStats integrateAdaptive(const RhsFn& f, State& u,
                                  double t0, double tEnd, double dt0,
                                  const AdaptiveRKOptions& opt) {
  // Bogacki-Shampine 3(2), FSAL. 3rd-order solution + 2nd-order embedded.
  const int orderLow = 2;            // embedded order -> error ~ dt^(orderLow+1)
  const double expo = 1.0 / (orderLow + 1.0); // = 1/3
  // PI controller exponents (Hairer & Wanner style):
  //   fac = safety * err^-alpha * errPrev^beta
  const double alpha = 0.7 * expo;
  const double beta = 0.4 * expo;

  AdaptiveRKStats st;
  st.tReached = t0;
  const size_t n = u.size();
  if (n == 0) return st;

  // work arrays, allocated once for the whole integration
  State k1(n), k2(n), k3(n), k4(n), tmp(n), uNew(n);

  double t = t0;
  double dt = std::min(dt0, opt.dtMax);
  double errPrev = 1.0; // error norm of the last ACCEPTED step (PI memory)

  // FSAL: evaluate k1 once up front; reuse k4 as next k1 on accepted steps.
  f(t, u, k1);
  ++st.rhsEvals;

  // loop until tEnd, with a relative round-off tolerance on the final time
  while (t < tEnd - 1e-15 * std::max(1.0, std::abs(tEnd))) {
    // shorten the last step so the integration lands exactly on tEnd
    if (t + dt > tEnd) dt = tEnd - t;

    // Stages 2 and 3.
    for (size_t i = 0; i < n; ++i) tmp[i] = u[i] + 0.5 * dt * k1[i];
    f(t + 0.5 * dt, tmp, k2);
    for (size_t i = 0; i < n; ++i) tmp[i] = u[i] + 0.75 * dt * k2[i];
    f(t + 0.75 * dt, tmp, k3);

    // 3rd-order candidate solution.
    for (size_t i = 0; i < n; ++i)
      uNew[i] = u[i] + dt * (2.0 / 9.0 * k1[i] + 1.0 / 3.0 * k2[i] +
                             4.0 / 9.0 * k3[i]);
    // FSAL stage at the end point.
    f(t + dt, uNew, k4);
    st.rhsEvals += 3;

    // Embedded 2nd-order solution and error = uNew - uEmb.
    // uEmb = u + dt(7/24 k1 + 1/4 k2 + 1/3 k3 + 1/8 k4)
    // Each component's error is divided by its tolerance scale
    //   sc = atol + rtol * max(|u|, |uNew|)
    // so r = e/sc = 1 means "exactly at tolerance".
    double errL2 = 0.0, errMax = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double uEmb = u[i] + dt * (7.0 / 24.0 * k1[i] + 0.25 * k2[i] +
                                       1.0 / 3.0 * k3[i] + 0.125 * k4[i]);
      const double e = uNew[i] - uEmb;
      const double sc = opt.atol + opt.rtol * std::max(std::abs(u[i]),
                                                       std::abs(uNew[i]));
      const double r = e / sc;
      errL2 += r * r;                             // accumulate for the RMS norm
      errMax = std::max(errMax, std::abs(r));     // max norm (diagnostic only)
    }
    errL2 = std::sqrt(errL2 / static_cast<double>(n)); // scaled RMS error
    // floor avoids pow(0, -alpha) = inf when the error is exactly zero
    const double errNorm = std::max(errL2, 1e-300);

    if (errNorm <= 1.0) {
      // Accept: advance time, adopt uNew, reuse k4 as the next k1.
      t += dt;
      u.swap(uNew);
      std::swap(k1, k4); // FSAL: k4(t+dt) becomes next k1
      ++st.accepted;
      st.tReached = t;
      st.dtLast = dt;
      st.maxNormLast = errMax;
      errPrev = errNorm;
    } else {
      // Reject: u and k1 are unchanged, so the retry reuses them as they are.
      ++st.rejected;
    }

    // PI step-size update (used after both accept and reject).
    double fac = opt.safety * std::pow(errNorm, -alpha) *
                 std::pow(errPrev, beta);
    fac = std::clamp(fac, opt.facMin, opt.facMax);  // limit change per step
    double dtNext = dt * fac;
    dtNext = std::min(dtNext, opt.dtMax);

    if (dtNext < opt.dtMin) {
      // Do NOT grind forever at a useless step: report and stop.
      st.aborted = true;
      st.dtLast = dtNext;
      break;
    }
    dt = dtNext;
  }

  return st;
}

} // namespace sgn
