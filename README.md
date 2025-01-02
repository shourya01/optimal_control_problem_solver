# OCPS Usage Documentation

## Overview

The **OCPS** class (short for *Optimal Control Problem Solver*) solves general continuous nonlinear programs of the form:

$$
\begin{aligned}
\text{minimize}_{x\in\mathbb{R}^n}\quad & f(x_1,\cdots,x_T) \\
\text{subject to}\quad & c_i(x_1,\cdots,x_T) = 0,\; i\in\mathcal{E}, \\
& c_j(x_1,\cdots,x_T) \le 0,\; j\in\mathcal{I}, \\
& x_{\min,t} \le x_t \le x_{\max,t}, \forall t \in [T]\\
& \text{Initial conditions specifiable.}
\end{aligned}
$$

Here, \(f\) is the objective, \(c_i\) are equality/inequality constraints, and \(x_{\min}, x_{\max}\) are variable bounds. You provide gradients and hessians for the objective and each constraint as a dictionary, thereby empowering you to define $f$ and $c$ arbitrary functions of the continuous variable $x$.

The solver builds on **Ipopt** via `cyipopt`. It allows you to:

- Define multiple continuous variables over timesteps (with bounds).
- Add general constraints, each with a custom sense (`eq` or `leq`).
- Provide exact gradients and Hessians.
- Solve the resulting problem and retrieve the solution.
- Prevent accidental asymmetric Hessian definitions by raising an error if both `(i, j)` and `(j, i)` are added.
- Provide sparse structures for Jacobian/Hessian lookup.
- Save the final solution (primal and dual) as a `.npz` file if desired.

## Defining a Problem

1. **Create an instance of `OCPS`:**

   ```python
   from optimal_control_problem_solver import OCPS

   prob = OCPS()
   ```

2. **Add Variables**:

   ```python
   prob._add_variable("x", vec_dim=1, time_steps=2, lb=0, ub=None)
   ```

   - This creates a variable named "x" with dimension (1) for (2) timesteps.
   - Bounds can be `None` (infinite), a scalar, or a matching numpy array.

3. **Indexing**:

   ```python
   i0 = prob._get("x", 0, 0)
   i1 = prob._get("x", 0, 1)
   ```

   - Gets global indices for each dimension/time of variable "x".

4. **Set Objective**:

   ```python
   def obj_val(x):
       return x[i0]**2 + x[i1]**2

   obj_grad = {
       i0: lambda x: 2*x[i0],
       i1: lambda x: 2*x[i1]
   }

   obj_hess = {
       (i0, i0): lambda x: 2.0,
       (i1, i1): lambda x: 2.0
   }

   prob._set_objective(obj_val, obj_grad, obj_hess)
   ```

   - `obj_val(x)` returns the scalar objective.
   - `obj_grad` is a dict from variable indices to partial derivatives.
   - `obj_hess` is a dict from `(i, j)` index pairs to second partial derivatives.
   - The class will raise an error if you define both `(i, j)` and `(j, i)` for the same entry.

5. **Add Constraints**:

   ```python
   # eq constraint: x0 = 3
   eq_func = lambda x: x[i0] - 3
   eq_jac = {i0: lambda x: 1.0}
   eq_hes = {}
   prob._add_constraint(eq_func, eq_jac, eq_hes, sense='eq', name='fix_x0')

   # leq constraint: x1 <= 2
   leq_func = lambda x: x[i1] - 2
   leq_jac = {i1: lambda x: 1.0}
   leq_hes = {}
   prob._add_constraint(leq_func, leq_jac, leq_hes, sense='leq', name='limit_x1')
   ```

   - `sense='eq'` sets `(g(x)=0)`.
   - `sense='leq'` sets `(g(x) <= 0)`.

6. **Finalize**:

   ```python
   prob.finalize_problem()
   ```

   - Builds sparse Jacobian/Hessian structures and calls the parent constructor from Ipopt.

## Example Problem

We are minimizing
$$
\min_{x_0, x_1} \quad x_0^2 + x_1^2
$$
subject to
$$
\begin{cases}
x_0 = 3 \quad (\text{an equality}),\\
x_1 \le 2 \quad (\text{an inequality}),\\
x_0 \ge 0,\, x_1 \ge 0 \quad (\text{bounds}).
\end{cases}
$$

In code:

```python
from optimal_control_problem_solver import OCPS
import numpy as np

prob = OCPS()
prob._add_variable("x", 1, 2, lb=0, ub=None)
i0 = prob._get("x", 0, 0)
i1 = prob._get("x", 0, 1)

def obj_val(x):
    return x[i0]**2 + x[i1]**2

obj_grad = {i0: lambda x: 2*x[i0], i1: lambda x: 2*x[i1]}
obj_hess = {(i0, i0): lambda x: 2.0, (i1, i1): lambda x: 2.0}
prob._set_objective(obj_val, obj_grad, obj_hess)

eq_func = lambda x: x[i0] - 3
eq_jac = {i0: lambda x: 1.0}
eq_hes = {}
prob._add_constraint(eq_func, eq_jac, eq_hes, sense='eq', name='fix_x0')

leq_func = lambda x: x[i1] - 2
leq_jac = {i1: lambda x: 1.0}
leq_hes = {}
prob._add_constraint(leq_func, leq_jac, leq_hes, sense='leq', name='limit_x1')

prob.finalize_problem()
sol = prob.solve(np.array([1.0, 1.0]))
print("Solution:", sol)

primal_dict = prob.get_optimal_primals()
dual_dict   = prob.get_optimal_duals()

print("Primals:", primal_dict)
print("Duals:", dual_dict)
```

## Second Example: Nonlinear Objective and Constraint

Here, we illustrate a more general nonlinear problem:

Objective:
$$
\min_{(x_0, x_1)}\; (x_0 - 2)^2 + 3\sin(x_1),
$$
subject to the nonlinear inequality constraint:
$$
x_0^2 - x_1 - 2 \le 0,
$$
plus any bounds you see fit (e.g., $x_0 >= 0$, $x_1$ unbounded).

Example code:

```python
from optimal_control_problem_solver import OCPS
import numpy as np

prob2 = OCPS()
# Suppose we add 1D var "z" with dimension=2, time_steps=1 => 2 scalars total
# We'll keep z[0] >= 0, z[1] unbounded => shape is (2,1).
lb_z = np.array([[0.0], [-1e20]])

prob2._add_variable("z", vec_dim=2, time_steps=1, lb=lb_z, ub=None)

i0 = prob2._get("z", 0, 0)
i1 = prob2._get("z", 1, 0)

def obj_val2(x):
    return (x[i0] - 2)**2 + 3*np.sin(x[i1])

obj_grad2 = {
    i0: lambda x: 2*(x[i0] - 2),
    i1: lambda x: 3*np.cos(x[i1])
}

obj_hess2 = {
    (i0, i0): lambda x: 2.0,
    (i1, i1): lambda x: -3.0*np.sin(x[i1])
}

prob2._set_objective(obj_val2, obj_grad2, obj_hess2)

# Constraint: g(x) = x0^2 - x1 - 2 <= 0
def c2_func(x):
    return x[i0]**2 - x[i1] - 2

c2_jac = {
    i0: lambda x: 2*x[i0],
    i1: lambda x: -1.0
}
c2_hes = {
    (i0, i0): lambda x: 2.0
}

prob2._add_constraint(c2_func, c2_jac, c2_hes, sense='leq', name='nonlinear_constr')

prob2.finalize_problem()
sol2 = prob2.solve(np.array([1.0, 0.0]))
print("Second Example Solution:", sol2)
print("Second Example Primals:", prob2.get_optimal_primals())
print("Second Example Duals:", prob2.get_optimal_duals())
```

## Checking Derivatives

You can call `check_derivatives(x0, second_order=True, print_all=False)` to let Ipopt compare your supplied derivatives to a finite-difference approximation. If `print_all=False`, it prints only the rows with significant mismatches.

**Late Binding Reminder**: If you define gradients or Hessians in a loop, you must capture the loop variable at definition time or you’ll risk incorrect derivatives. For instance, use `lambda x, j=j: ...` in your dict entries.

## Saving the Solution

We provide `save_solution(filename="solution.npz")`, which stores all the primal variables and dual multipliers in a `.npz` file. For each variable named `foo`, it saves a key `"var_foo"`. For each constraint named `bar`, it saves a key `"dual_bar"`. Example usage:

```python
prob.save_solution("my_solution.npz")
```

Then, `np.load("my_solution.npz")` returns a dictionary with keys like `"var_x"` and `"dual_nonlinear_constr"` etc.

## Credits and License

Authored by Shourya Bose, referencing IPOPT and cyipopt documentation. AI tools were used to aid with writing this code.

Released under the MIT License.