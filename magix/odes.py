# Core functionality
# Authors: Thomas A. Scott https://www.scott-aero.com/

from functools import partial, reduce
from dataclasses import dataclass, field, is_dataclass, make_dataclass
from dataclasses import fields as get_fields

import scipy
import numpy as np
import jax
import jax.numpy as jnp
import jax.typing as jtp

# Forward Euler scheme
def step_fe(
    z_dyn, z, dmap_z_I, dmap_dz_I,
    dt, frizz_dyn,
):
    return z_dyn.at[...,dmap_z_I].add(dt*z_dyn[...,dmap_dz_I])

# Generic integrator step, currently set up for predetermind time steps
def integrator_step(i, args, fstep, frizz_dyn):
    t, z, dmap_z_I, dmap_dz_I, z_dyn, z_dyn_stack = args
    dt = t[i] - t[i-1]

    # Calculate derivative at last time and record, i.e. the state at i-1
    z_dyn = frizz_dyn(z_dyn=z_dyn, z=z)
    z_dyn_stack = z_dyn_stack.at[i-1,...].set(z_dyn)
    
    # Call integrator to progress independent variables from i-1 to i
    z_dyn = fstep(z_dyn, z, dmap_z_I, dmap_dz_I, dt, frizz_dyn)

    return t, z, dmap_z_I, dmap_dz_I, z_dyn, z_dyn_stack

# Integrator that takes a rizzy dynamics function
def setup_rizzinator(z, dmap_z_I, dmap_dz_I, fstep, frizz_dyn):
    # ODE Integrator function
    # TODO: this should be vmap or pmap outside, adjacent memory means doing each individually
    # Optimizing the NN should have batches as the inner dim (time as outer) but need to not copy s every time...

    # TODO: get deriv at time 0 for completeness? storing dx needs to be out of sync with x!
    _integrator_step = jax.jit(partial(integrator_step, fstep=fstep, frizz_dyn=frizz_dyn))
    
    def _integrator(z_dyn0, z, dt, T, _integrator_step=_integrator_step, dmap_z_I=dmap_z_I, dmap_dz_I=dmap_dz_I):
        Nt = jnp.ceil(T / dt).astype(int)
        t = (jnp.arange(Nt)*dt).at[-1].set(T)
        
        z_dyn_stack = jnp.zeros((Nt,)+z_dyn0.shape)
        _, _, _, _, z_dyn, z_dyn_stack = jax.lax.fori_loop(1, Nt, _integrator_step, (t, z, dmap_z_I, dmap_dz_I, z_dyn0, z_dyn_stack))
        
        # Final state in general only has independent variables at final time after exiting integrator, call dynamics once more to update to full state at final time
        z_dyn = frizz_dyn(z_dyn=z_dyn, z=z)
        z_dyn_stack = z_dyn_stack.at[-1,...].set(z_dyn)
        
        return t, z_dyn_stack
    
    return _integrator
