'''Asymptotic Stability Analysis Post Processing'''

import numpy as np
import matplotlib.pyplot as plt

beam_eigenvalues_trim = np.loadtxt('./output/llta_modes_aerodynamics/beam_modal_analysis/eigenvalues.dat')
sys_eigenvalues_trim = np.loadtxt('./output/llta_modes_aerodynamics/stability/aeroelastic_eigenvalues.dat')

# Flight modes
flight_modes = plt.figure()
plt.scatter(sys_eigenvalues_trim[:, 0], sys_eigenvalues_trim[:, 1],
           marker='x',
           color='k')
plt.xlim(-2, 0.5)
plt.ylim(-100, 100)
plt.grid()
plt.xlabel('Real Part, $Re(\lambda)$ [rad/s]')
plt.ylabel('Imaginary Part, $Im(\lambda)$ [rad/s]')
plt.show()

# Natural modes
natural_modes = plt.figure()
print(beam_eigenvalues_trim)
plt.scatter(np.zeros(len(beam_eigenvalues_trim)), beam_eigenvalues_trim,
           marker='x',
           color='k')
plt.xlim(-2, 6)
plt.ylim(-400, 400)
plt.grid()
plt.xlabel('Real Part, $Re(\lambda)$ [rad/s]')
plt.ylabel('Imaginary Part, $Im(\lambda)$ [rad/s]')
plt.show()

# Structural modes
structural_modes = plt.figure()
plt.scatter(sys_eigenvalues_trim[:, 0], sys_eigenvalues_trim[:, 1],
           marker='x',
           color='k')
plt.xlim(-2, 6)
plt.ylim(-400, 400)
plt.grid()
plt.xlabel('Real Part, $Re(\lambda)$ [rad/s]')
plt.ylabel('Imaginary Part, $Im(\lambda)$ [rad/s]')
plt.show()