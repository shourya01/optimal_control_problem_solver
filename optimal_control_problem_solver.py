import numpy as np
import uuid
from cyipopt import Problem

VERY_LARGE_NUMBER = 1e20

class OCPS(Problem):
    def __init__(self):
        # Variables
        self._index_map = {}  # (var_name, d, t) -> global index
        self._vars_info = {}  # var_name -> {dim, time_steps, indices=[]}
        self._num_vars = 0
        self._x_lb = []
        self._x_ub = []

        # Constraints
        self._constraints_funcs = []
        self._constraints_jac = []
        self._constraints_hes = []
        self._con_lb = []
        self._con_ub = []
        self._constraint_name_to_index = {}

        # Objective
        self._objective_func = lambda x: 0.0 
        self._objective_grad = {}
        self._objective_hessian = {}

        # Sparse structures for Jacobian/Hessian
        self._jac_rows = []
        self._jac_cols = []
        self._hes_rows = []
        self._hes_cols = []

        # Solve status
        self._last_solution = None

    def _resolve_bound_array(self, value, shape, default):
        if value is None:
            return np.full(shape, default, dtype=float)
        elif isinstance(value, (int, float)):
            return np.full(shape, value, dtype=float)
        elif isinstance(value, np.ndarray):
            if value.shape != shape:
                raise ValueError(f"Expected shape {shape}, got {value.shape}")
            return value.astype(float)
        else:
            raise TypeError("lb/ub must be None, a scalar, or a numpy array.")

    def _add_variable(self, var_name, vec_dim, time_steps, lb=None, ub=None):
        shape = (vec_dim, time_steps)
        lb_arr = self._resolve_bound_array(lb, shape, -VERY_LARGE_NUMBER)
        ub_arr = self._resolve_bound_array(ub, shape, VERY_LARGE_NUMBER)

        var_indices = []
        for t in range(time_steps):
            for d in range(vec_dim):
                idx = self._num_vars
                self._index_map[(var_name, d, t)] = idx
                self._num_vars += 1
                self._x_lb.append(lb_arr[d, t])
                self._x_ub.append(ub_arr[d, t])
                var_indices.append(idx)

        self._vars_info[var_name] = {
            "dim": vec_dim,
            "time_steps": time_steps,
            "indices": var_indices
        }

    def _get(self, var_name, d, t):
        return self._index_map[(var_name, d, t)]

    def _check_asymmetric_hessian_entries(self, hess_dict, is_objective=False, constraint_name=None):
        """
        Ensures that if (i, j) is present, (j, i) is NOT separately present
        with a different function. If both appear, we raise an error.
        """
        for (i, j) in hess_dict.keys():
            if (j, i) in hess_dict and (j, i) != (i, j):
                # We found both (i, j) and (j, i). That's disallowed.
                if is_objective:
                    raise ValueError(
                        f"Asymmetric Hessian entries provided for objective: "
                        f"both ({i}, {j}) and ({j}, {i}) exist."
                    )
                else:
                    name_str = constraint_name if constraint_name else "unknown"
                    raise ValueError(
                        f"Asymmetric Hessian entries provided for constraint '{name_str}': "
                        f"both ({i}, {j}) and ({j}, {i}) exist."
                    )

    def _set_objective(self, val_func, grad_dict, hess_dict):
        # Check for asymmetry
        self._check_asymmetric_hessian_entries(hess_dict, is_objective=True)

        self._objective_func = val_func
        self._objective_grad = grad_dict
        self._objective_hessian = hess_dict

    def _add_constraint(self, func, jac_dict, hess_dict, sense='eq', name=None):
        if sense not in ('eq', 'leq'):
            raise ValueError("sense must be 'eq' or 'leq'")
        if not name:
            name = str(uuid.uuid4())[:8]

        # Check for asymmetry
        self._check_asymmetric_hessian_entries(hess_dict, is_objective=False, constraint_name=name)

        idx = len(self._constraints_funcs)
        self._constraints_funcs.append(func)
        self._constraints_jac.append(jac_dict)
        self._constraints_hes.append(hess_dict)
        self._constraint_name_to_index[name] = idx

        if sense == 'eq':
            self._con_lb.append(0.0)
            self._con_ub.append(0.0)
        else:  # 'leq'
            self._con_lb.append(-VERY_LARGE_NUMBER)
            self._con_ub.append(0.0)

    def finalize_problem(self):
        # Print a summary of problem size
        print(f"Setting up a problem with {self._num_vars} variables "
            f"and {len(self._constraints_funcs)} constraints.")

        # Count & print objective Jacobian/Hessian nonzeros
        obj_jac_keys = list(self._objective_grad.keys())
        obj_hes_keys = list(self._objective_hessian.keys())
        print(f"  Objective Jacobian nonzeros: {len(obj_jac_keys)}")
        print(f"  Objective Hessian nonzeros:  {len(obj_hes_keys)}")

        # Count & print constraints Jacobian/Hessian nonzeros
        con_jac_count = 0
        con_hes_count = 0
        for jac_dict in self._constraints_jac:
            con_jac_count += len(jac_dict)
        for hes_dict in self._constraints_hes:
            con_hes_count += len(hes_dict)

        print(f"  Constraints Jacobian nonzeros: {con_jac_count}")
        print(f"  Constraints Hessian nonzeros:  {con_hes_count}")

        # Convert constraints bounds
        cl = np.array(self._con_lb)
        cu = np.array(self._con_ub)

        # Convert variable bounds
        lb_arr = np.array(self._x_lb)
        ub_arr = np.array(self._x_ub)

        # Build the sparse Jacobian structure
        jac_struct = []
        for c_idx, jac_dict in enumerate(self._constraints_jac):
            for var_idx in jac_dict.keys():
                jac_struct.append((c_idx, var_idx))
        jac_struct.sort(key=lambda x: (x[0], x[1]))
        self._jac_rows = [r for (r, _) in jac_struct]
        self._jac_cols = [c for (_, c) in jac_struct]

        # Build the sparse Hessian structure
        hes_pairs = set()
        for (i, j) in self._objective_hessian.keys():
            i2, j2 = (i, j) if i >= j else (j, i)
            hes_pairs.add((i2, j2))
        for hdict in self._constraints_hes:
            for (i, j) in hdict.keys():
                i2, j2 = (i, j) if i >= j else (j, i)
                hes_pairs.add((i2, j2))

        hes_list = sorted(list(hes_pairs), key=lambda x: (x[0], x[1]))
        self._hes_rows = [r for (r, _) in hes_list]
        self._hes_cols = [c for (_, c) in hes_list]

        # Call the parent initializer
        super().__init__(
            n=self._num_vars,
            m=len(self._constraints_funcs),
            cl=cl,
            cu=cu,
            lb=lb_arr,
            ub=ub_arr
        )
        self.add_option("hessian_approximation", "exact")

        # Set a high number for maximum iterations
        self.add_option("max_iter", 1000000)

        # # Tighten constraint tolerance
        # self.add_option("constr_viol_tol", 1e-10)

    # -------------------- Ipopt Required Callbacks --------------------
    def objective(self, x):
        return self._objective_func(x)

    def gradient(self, x):
        grad = np.zeros(self._num_vars)
        for i, gf in self._objective_grad.items():
            grad[i] = gf(x)
        return grad

    def constraints(self, x):
        return np.array([f(x) for f in self._constraints_funcs])

    def jacobian(self, x):
        # Return a 1D array of partial derivatives in the same order as jacobianstructure
        vals = []
        for (row, col) in zip(self._jac_rows, self._jac_cols):
            fn = self._constraints_jac[row].get(col, None)
            vals.append(fn(x) if fn else 0.0)
        return np.array(vals, dtype=float)

    def jacobianstructure(self):
        # Return the row, col lists built in finalize_problem
        return (self._jac_rows, self._jac_cols)

    def hessian(self, x, lagrange, obj_factor):
        # Combine objective Hessian and constraints Hessians into a dict
        Htmp = {}

        # Objective contributions
        for (i, j), hf in self._objective_hessian.items():
            i2, j2 = (i, j) if i >= j else (j, i)
            Htmp[(i2, j2)] = Htmp.get((i2, j2), 0.0) + obj_factor * hf(x)

        # Constraint contributions
        for c_idx, hess_dict in enumerate(self._constraints_hes):
            lam = lagrange[c_idx]
            for (i, j), hf in hess_dict.items():
                i2, j2 = (i, j) if i >= j else (j, i)
                Htmp[(i2, j2)] = Htmp.get((i2, j2), 0.0) + lam * hf(x)

        # Gather in order of self._hes_rows, self._hes_cols
        vals = []
        for (r, c) in zip(self._hes_rows, self._hes_cols):
            vals.append(Htmp.get((r, c), 0.0))

        return np.array(vals, dtype=float)

    def hessianstructure(self):
        # Return the row, col lists for the Hessian built in finalize_problem
        return (self._hes_rows, self._hes_cols)

    # -------------------- Extended Methods --------------------
    def solve(self, x0):
        sol = super().solve(x0)
        self._last_solution = sol
        return sol

    def get_optimal_primals(self):
        if self._last_solution is None:
            raise RuntimeError("Problem not solved yet.")
        status = self._last_solution[1]['status']
        if status != 0:
            raise RuntimeError(f"Problem solve failed, status code={status}.")

        x_opt = self._last_solution[0]
        out = {}
        for var_name, info in self._vars_info.items():
            dim = info["dim"]
            tsteps = info["time_steps"]
            indices = info["indices"]
            arr = x_opt[indices].reshape((dim, tsteps), order='C')
            out[var_name] = arr
        return out

    def get_optimal_duals(self):
        if self._last_solution is None:
            raise RuntimeError("Problem not solved yet.")
        status = self._last_solution[1]['status']
        if status != 0:
            raise RuntimeError(f"Problem solve failed, status code={status}.")

        mult_g = self._last_solution[1]['mult_g']
        out = {}
        for cname, cidx in self._constraint_name_to_index.items():
            out[cname] = mult_g[cidx]
        return out
    
    def get_optimal_residuals(self):
        if self._last_solution is None:
            raise RuntimeError("Problem not solved yet.")
        status = self._last_solution[1]['status']
        if status != 0:
            raise RuntimeError(f"Problem solve failed, status code={status}.")

        x_opt = self._last_solution[0]
        g_vals = self.constraints(x_opt)  # array of constraint values

        out = {}
        for cname, cidx in self._constraint_name_to_index.items():
            out[cname] = g_vals[cidx]

        return out

    def check_derivatives(self, x0, second_order=True, print_all=False):
        old_deriv_test = None
        old_deriv_test_print_all = None
        try:
            old_deriv_test = self.get_option("derivative_test")
            old_deriv_test_print_all = self.get_option("derivative_test_print_all")
        except:
            pass

        # Pick first-order or second-order derivative_test
        self.add_option("derivative_test", "second-order" if second_order else "first-order")
        # If print_all=False => "no", so Ipopt only prints rows with mismatches
        self.add_option("derivative_test_print_all", "yes" if print_all else "no")

        sol = self.solve(x0)

        # Restore original settings
        if old_deriv_test is not None:
            self.add_option("derivative_test", old_deriv_test)
        if old_deriv_test_print_all is not None:
            self.add_option("derivative_test_print_all", old_deriv_test_print_all)

        return sol

    def save_solution(self, filename="solution.npz"):
        """
        Replacement for save_solution that always uses self._last_solution,
        and reshapes with order='F' to stay consistent with the (t outer, d inner)
        loop in _add_variable.

        Saved .npz structure:
        raw_x : array(...)         -> the raw solution
        var_<var_name> : array(...) -> shape (dim, time_steps), unflattened with order='F'
        dual_<constraint_name> : array([multiplier])
        res_<constraint_name> : array([residual])
        """
        import numpy as np

        if self._last_solution is None:
            raise RuntimeError("No solution available; call solve() before saving.")

        status = self._last_solution[1]['status']
        if status != 0:
            raise RuntimeError(f"Problem solve failed, status code={status}.")

        x_opt = self._last_solution[0]
        data_to_save = {}
        data_to_save["raw_x"] = x_opt

        # Helper to reshape in 'F' order, matching the enumeration of (t outer, d inner)
        def unflatten_primal(x_array):
            result = {}
            for var_name, info in self._vars_info.items():
                indices = info["indices"]
                dim = info["dim"]
                tsteps = info["time_steps"]
                arr_2d = x_array[indices].reshape((dim, tsteps), order='F')
                result[var_name] = arr_2d
            return result

        # Unflatten primal variables
        x_dict = unflatten_primal(x_opt)

        # Collect dual multipliers
        lam_dict = self.get_optimal_duals()

        # Compute residuals
        g_vals = self.constraints(x_opt)

        # Save primal arrays
        for var_name, arr_2d in x_dict.items():
            data_to_save[f"var_{var_name}"] = arr_2d

        # Save duals and residuals
        for cname, cidx in self._constraint_name_to_index.items():
            data_to_save[f"dual_{cname}"] = np.array([lam_dict[cname]])
            data_to_save[f"res_{cname}"] = np.array([g_vals[cidx]])

        np.savez(filename, **data_to_save)
        print(f"Saved solution to {filename}.")
