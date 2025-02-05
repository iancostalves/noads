from jax.numpy import array

from core.models.util import InterpolatedUnivariateSpline


class AircraftTechParameter:
    name: str

    value_2020: float

    value_2040: float

    value_2060: float

    def __init__(self, name: str, values: tuple[float, float, float]):
        self.name = name
        self.value_2020 = values[0]
        self.value_2040 = values[1]
        self.value_2060 = values[2]

    def value_at_entry_into_service(self, entry_into_service):
        x_data = array([2020, 2040, 2060])
        y_data = array([self.value_2020, self.value_2040, self.value_2060])
        spline = InterpolatedUnivariateSpline(x_data, y_data, k=2)
        return spline(entry_into_service)
