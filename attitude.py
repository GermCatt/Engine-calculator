import pyquaternion
import numpy as np

"""
Prelude.

Data format:
    Accelerations - [ax, ay, az]
    Angular rated - [wx, wy, wz]

g is taken as 9.81 m/s^2

Accelerations are calculated in aircraft's coordinates
Clean accelerations mean that these are taken without the incluence of g.

KLcpb
"""


class Attitude():
    def __init__(self, delta_t) -> None:
        self.dt = delta_t

    def calculate(self, timesteps, accs, omegas):
        """
        Timesteps = a list of durations of each step
        Accs = accelerations [numpy.ndarray]
        Omegas = angular rates [numpy.ndarray]


        This function calculates two series of values:
            g in aircraft's coordinates
            Clean accelerations in aircraft's coordinates
        """
        if not len(accs) == len(omegas):
            raise AttributeError(f"Lengths of accs and omegas should be the same. Got {len(accs) and {len(omegas)} }")

        if not type(accs[0]) == np.ndarray or not type(omegas[0]) == np.ndarray:
            raise AttributeError(
                f"Type of input data should be numpy.ndarray! Got types {type(accs)} and {type(omegas)}")

        # getting initial angular rates and accelerations
        omega0 = omegas[0]
        a0 = accs[0]
        g_quaternion = pyquaternion.Quaternion(0, -a0[0], -a0[1], -a0[2])  # initial acceleration in form of quaternion

        # creating arrays for recording accelerstions and gs
        self.gs = []
        self.clean_accs = []
        q = np.array([1, 0, 0, 0])  # initial position
        for i in range(len(omegas)):
            omega = omegas[i] - omega0  # to remove gyroscope drift
            wx = omega[0]
            wy = omega[1]
            wz = omega[2]
            # dt = self.dt
            dt = timesteps[i]
            W = np.array([[2 / dt, -wx, -wy, -wz],  # taken from https://mariogc.com/post/angular-velocity-quaternions/
                          [wx, 2 / dt, wz, -wy],
                          [wy, -wz, 2 / dt, wx],
                          [wz, wy, -wx, 2 / dt]])

            q = dt / 2 * np.dot(W, q)  # from article above
            q = q / np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)  # normalizing the quaternion
            quat = pyquaternion.Quaternion(q)
            g_rotated = quat.inverse * g_quaternion * quat  # rotating g to get it in aircraft's coordinates
            g_rotated_vector = g_rotated.imaginary  # getting rotated g in form of vector
            self.gs.append(g_rotated_vector)

            # calculating clean accelerations
            clean_acc = accs[i] + g_rotated_vector
            self.clean_accs.append(clean_acc)

    def get_gs(self, ):
        return self.gs

    def get_accs(self, ):
        return self.clean_accs



