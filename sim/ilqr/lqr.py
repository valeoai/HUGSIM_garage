from dataclasses import replace

from sim.ilqr.lqr_solver import ILQRSolverParameters, ILQRWarmStartParameters, ILQRSolver
import numpy as np

solver_params = ILQRSolverParameters(
    discretization_time=0.5,
    state_cost_diagonal_entries=[1.0, 1.0, 10.0, 0.0, 0.0],
    input_cost_diagonal_entries=[1.0, 10.0],
    state_trust_region_entries=[1.0] * 5,
    input_trust_region_entries=[1.0] * 2,
    max_ilqr_iterations=100,
    convergence_threshold=1e-6,
    max_solve_time=0.05,
    max_acceleration=3.0,
    max_steering_angle=np.pi / 3.0,
    max_steering_angle_rate=0.4,
    min_velocity_linearization=0.01,
    wheelbase=2.7
)

warm_start_params = ILQRWarmStartParameters(
    k_velocity_error_feedback=0.5,
    k_steering_angle_error_feedback=0.05,
    lookahead_distance_lateral_error=15.0,
    k_lateral_error=0.1,
    jerk_penalty_warm_start_fit=1e-4,
    curvature_rate_penalty_warm_start_fit=1e-2,
)

lqr = ILQRSolver(solver_params=solver_params, warm_start_params=warm_start_params)

# `discretization_time` is used by the solver for three things at once: the spacing it assumes
# between the reference poses, the integration step of its rollout, and hence the interval over
# which the returned first input is meant to be held. It therefore has to match the interval the
# caller actually applies the command for. Solvers are cached per discretization time; they carry
# no state between solve() calls.
_solvers = {solver_params.discretization_time: lqr}


def set_max_solve_time(seconds):
    """The wall-clock budget of one solve, or None to stop on iterations and convergence alone.

    The shipped 0.05 s (nuPlan's) makes the number of iterations, hence the command, depend on
    how busy the machine is: the same plan from the same state can be tracked differently from
    one run to the next. None makes the tracker a function of its inputs. Cached solvers are
    dropped, so the next solve uses the new budget.
    """
    global solver_params, lqr
    solver_params = replace(solver_params, max_solve_time=None if seconds is None else float(seconds))
    _solvers.clear()
    lqr = _get_solver(solver_params.discretization_time)  # the module-level solver follows too


def _get_solver(discretization_time):
    if discretization_time not in _solvers:
        _solvers[discretization_time] = ILQRSolver(
            solver_params=replace(solver_params, discretization_time=discretization_time),
            warm_start_params=warm_start_params,
        )
    return _solvers[discretization_time]


def plan2control(plan_traj, init_state, discretization_time=None):
    """
    Args:
        plan_traj: reference states (N, [x, y, heading, velocity, steering_angle]), where
            consecutive rows are `discretization_time` apart and row 0 is the current state.
        init_state: current state, of the same 5-element layout.
        discretization_time: interval the returned command will be applied for. Defaults to the
            module-level solver parameter.
    """
    if discretization_time is None:
        discretization_time = solver_params.discretization_time
    current_state = init_state
    solutions = _get_solver(discretization_time).solve(current_state, plan_traj)
    optimal_inputs = solutions[-1].input_trajectory
    accel_cmd = optimal_inputs[0, 0]
    steering_rate_cmd = optimal_inputs[0, 1]
    return accel_cmd, steering_rate_cmd

if __name__ == '__main__':
    # plan_traj = np.zeros((6,5))
    # plan_traj[:, 0] = 1
    # plan_traj[:, 1] = np.ones(6)
    # plan_traj = np.cumsum(plan_traj, axis=0)
    # print(plan_traj)
    plan_traj = np.array([[-0.18724936,  2.29100776,  0.,          0.,          0.,        ],
                        [-0.29260731,  2.2971828 ,  0.,          0.,          0.        ],
                        [-0.46831554,  2.55596018,  0.,          0.,          0.        ],
                        [-0.5859955 ,  2.73183298,  0.,          0.,          0.        ],
                        [-0.62684   ,  2.84659386,  0.,          0.,          0.        ],
                        [-0.67761713,  2.80647802,  0.,          0.,          0.        ]])
    plan_traj = plan_traj[:, [1,0,2,3,4]]
    init_state = np.array([0.00000000e+00, 3.46944695e-17, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
    print(plan_traj.shape, init_state.shape)
    acc, steer = plan2control(plan_traj, init_state)
    print(acc, steer)
