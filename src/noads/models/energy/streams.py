
class Stream:
    name: str
    unit: str

    def __init__(self, name: str, unit: str):
        self.name = name
        self.unit = unit


class Impact(Stream):
    budget: float

    def __init__(self, name, unit, budget: float):
        self.budget = budget
        super().__init__(name=name, unit=unit)


class MaterialStream(Stream):
    density: float
    unit = "kg"

    def __init__(self, name, density: float):
        self.density = density
        super().__init__(name=name, unit="kg")

    def volume_to_mass(self, volume):
        return volume * self.density

    def mass_to_volume(self, mass):
        return mass / self.density
