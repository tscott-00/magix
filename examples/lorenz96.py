import os
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import magix
from magix import magixmethod, magiclass, Placeholder, Dynamic, Derivative
import magix.odes



@magiclass
class Lorenz96:
    x:  jax.Array = Placeholder()
    dx: jax.Array = Derivative('x')

    @classmethod
    def new(cls, dims):
        return cls(x=Dynamic(dims))

    @magixmethod
    def __call__(self):
        # dx_i/dt = (x_{i+1} - x_{i-2}})x_{i-1} + 8, with indices wrapping around when under or overflowing
        self.dx = (jnp.roll(self.x,-1,axis=-1) - jnp.roll(self.x,2,axis=-1))*jnp.roll(self.x,1,axis=-1) - self.x + 8.0

if __name__ == "__main__":
    z_meta = magix.bake_tree(Lorenz96.new(dims=8))
    integrator = magix.odes.make_integrator(z_meta, magix.odes.step_fe)

    # Allocate state with state and derivatives starting at zeros then initialize x to 1.0
    z0 = magix.zeros(z_meta)
    z0.x = 8.0
    z0.x[0] += 0.01 # Slight proturbation

    # Run JIT compiled integrator
    t, z = integrator(z0, dt=1E-2, T=30.0)
    # Print state at final time
    print(z[-1].x)

    # TODO: Plot first 3 states
