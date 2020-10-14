from gpflow.params import Parameter, Parameterized
from abc import abstractmethod
from gpflow import decors
import tensorflow as tf
from gpflow import settings


class InducingPoints(Parameterized):
    """
    Real-space inducing points
    """

    def __init__(self, Z):
        """
        :param Z: the initial positions of the inducing points, size M x D
        """
        super().__init__()
        self.Z = Parameter(Z, dtype=settings.float_type)

    def __len__(self):
        return self.Z.shape[0]

    @decors.params_as_tensors
    def Kuu(self, kern, jitter=0.0):
        Kzz = kern.K(self.Z)
        Kzz += jitter * tf.eye(len(self), dtype=settings.dtypes.float_type)
        return Kzz

    @decors.params_as_tensors
    def Kuf(self, kern, Xnew):
        Kzx = kern.K(self.Z, Xnew)
        return Kzx

    Kzz = Kuu
    Kzx = Kuf
