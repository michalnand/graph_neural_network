import numpy

class Randomizer:
    def __init__(self, dims = 3, sigma_position = 1.0, sigma_velocity = 1.0):
        self.dims               = dims
        self.sigma_position     = sigma_position
        self.sigma_velocity     = sigma_velocity

        self.next()
  
    def next(self):
        self.position_offset = self.sigma_position*numpy.random.randn(self.dims)
        self.velocity_offset = self.sigma_velocity*numpy.random.randn(self.dims)

    def get_position(self):
        return self.sigma_position*numpy.random.randn(self.dims) + self.position_offset

    def get_velocity(self):
        return self.sigma_velocity*numpy.random.randn(self.dims) + self.velocity_offset

    def get_scale(self, min = 0.1, max = 1.0):
        return numpy.random.rand()*(max - min) + min