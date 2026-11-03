# ADR-0007: What the paper's IK "1e-5 threshold" actually controls

- Status: accepted
- Date: 2026-09-20
- Tags: `[PAPER]` publishes three numbers, `[OPEN]` what they mean

## Context

Paper appendix A.4 gives the base-search IK settings as:

> the mink IK solver, quadprog backend, 100 iterations, `1e-5` threshold

Three concrete questions follow, none of which the paper answers:

1. What does the `1e-5` measure — the task error (metres / radians), the
   per-iteration velocity, or a solver primal tolerance?
2. What counts as "feasible" for the feasibility rate `FR` in equation 8?
3. Are there orientation weights, damping, or a timestep?

Inspection of `mink` 1.3.0 shows that the library has **no iteration count and no
tolerance parameter at all**:

* `mink.solve_ik(configuration, tasks, dt, solver, damping=..., limits=...)`
  returns a *velocity*, which the caller integrates. There is no `options={...}`
  dict and no `max_iter`/`tol` argument.
* Iteration limits and tolerances, where they exist, belong to the QP backend and
  are forwarded through `qpsolvers` kwargs: DAQP takes `iter_limit` (default
  10000) and `primal_tol` (1e-6); OSQP takes `max_iter`/`eps_abs`/`eps_rel`;
  quadprog exposes only `factorized`.

So "100 iterations, 1e-5 threshold" cannot be a solver configuration. It has to
be an **outer loop**.

## Decision

`ego2robot.robots.franka.PandaIKSolver` implements the outer loop explicitly:

```text
for iteration in 1..iterations:            # default 100, from config
    velocity = mink.solve_ik(config, [task], dt=1e-2, solver=<backend>, damping=1e-5, limits=[ConfigurationLimit])
    config.integrate_inplace(velocity, dt=1e-2)
    if ||position_error|| < tolerance and ||orientation_error|| < tolerance: converged
```

* `iterations` and `tolerance` come from `robot_binding.ik` in the profile
  (`100`, `1e-5`), i.e. the paper's published numbers, interpreted as the outer
  loop's budget and stopping criterion.
* `tolerance` is applied to the **task error** reported by Mink's `FrameTask`
  (metres for the translation block, radians for the rotation block).
* `ConfigurationLimit` is always passed, so joint limits are enforced by the QP
  rather than checked after the fact.
* A frame counts as successful only when the solver converged **and** the
  FK re-derivation of `T_B_TCP` is within `1e-4 m` and `1e-3 rad` of the target.
  This is deliberately looser than `tolerance`, because the solver stops on its
  task error while the acceptance test re-derives the pose through a full forward
  kinematics step.
* "Feasible" for the base-search feasibility rate is the same predicate. The
  paper's equation 8 uses `1[IK(...) is feasible]` without defining it; using the
  same predicate as the trajectory gate is what makes the two comparable.

Seed policy (not in the paper): `q[t-1]` seeds `q[t]`
(`docs/system-design.md` 8.1), and when that fails, up to
`robot_binding.ik.seed_attempts` deterministic alternate seeds
(`ego2robot.robots.stage._seed_variant`) are tried and the lowest-position-error
result is kept. This exists because the Panda has genuine multiple branches and
a wrong branch can be locally optimal.

## Consequences

* The IK cost matches the paper's numbers but the *semantics* are this project's.
  A different implementation reading "1e-5" as a primal tolerance would produce
  different trajectories.
* `raw`/`filtered`/`smoothed` trajectories and the per-frame `solver_status`,
  `ik_iterations` and closure errors are all persisted, so the effect of these
  choices is measurable rather than asserted.
* On the five validation cases: IK success 100 %, p95 FK position error
  `< 1e-5 m`, far inside the MVP gate of 1 cm.

## Evidence

* `ego2robot/robots/franka/adapter.py` (`PandaIKSolver.solve`)
* `ego2robot/robots/stage.py` (per-frame loop, seed recovery, metrics)
* `configs/pipeline/mvp.yaml` (`robot_binding.ik`)
* `docs/references/panda-mjcf-and-mink-research.md` (mink 1.3.0 API surface)

## Alternatives considered

* **Treat 100 iterations as a QP iteration limit** — not expressible: quadprog
  has no such parameter, and DAQP's default is 10000.
* **Stop when `||velocity|| * dt < 1e-5`** — this is what the third-party
  reproduction does. Rejected as the primary criterion because a velocity
  threshold is scale-dependent (it mixes metres and radians through a joint-space
  norm), whereas a task-error threshold is directly the quantity the gate checks.
  It is worth revisiting if IK convergence turns out to be the bottleneck.
* **Use `daqp` instead of `quadprog`** — not chosen: the paper names quadprog and
  `quadprog` has a working cp313 wheel. `daqp` is available as a fallback via
  `robot_binding.ik.qp_backend` if quadprog ever fails to converge.
