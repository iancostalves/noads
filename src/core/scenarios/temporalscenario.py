from collections.abc import Mapping
from collections.abc import Sequence

from diffrax import AbstractAdjoint
from diffrax import BacksolveAdjoint
from diffrax import DirectAdjoint
from diffrax import RecursiveCheckpointAdjoint
from diffrax import diffeqsolve
from diffrax import Dopri5
from diffrax import ODETerm
from diffrax import SaveAt
from gemseo.typing import RealArray
from jax import vmap
from jax.numpy import array
from jax.numpy import ones
from jax.numpy import interp
from jax.numpy import sum
from jax.numpy import where
from jax.scipy.integrate import trapezoid
from numpy import append

from numpy import arange
from strenum import StrEnum

from core.model import Model
from core.models.util import InterpolatedUnivariateSpline
from gemseo_jax.jax_chain import JAXChain
from gemseo_jax.jax_discipline import DataType
from gemseo_jax.jax_discipline import JAXDiscipline


def delay1_rhs(t, y, args):
    """Right-hand side function of a 1st order delay."""
    delay_state = y
    time_constant, control_values, control_times = args
    input_value = interpolate_data(t, control_times, control_values)
    delay_rate = (input_value - delay_state) / time_constant
    return delay_rate


class TemporalScenario(Model):
    class AdjointMethod(StrEnum):
        """The method to compute the Adjoint of control variables."""

        RECURSIVE = "RecursiveCheckpoint"
        DIRECT = "Direct"
        BACKSOLVE = "Backsolve"

    _integration_prefix: str = "cumulative"

    _constant_prefix: str = "constant"

    _control_prefix: str = "control"

    _interpolation_prefix: str = "interpolate"

    _constraint_prefix: str = "constraint"

    __adjoint: AbstractAdjoint

    models: list[Model]

    constant_inputs: list[str]

    interpolated_inputs: list[str]

    control_delay_times: Mapping[str, float]

    constrained_control_groups: Mapping[str: Sequence[str]]

    custom_controls: Sequence[str]

    non_modified_inputs: list[str]

    time_integrated_outputs: list[str]

    time_vector: RealArray

    time_interpolation: RealArray

    cubic_interpolation: bool

    vectorized_chain: JAXChain

    unvectorized_chain: JAXChain

    def __init__(
        self,
        name: str,
        models: Sequence[Model],
        constant_inputs: Sequence[str],
        control_delay_times: Mapping[str, float],
        constrained_control_groups: Mapping[str: Sequence[str]],
        custom_controls: Sequence[str],
        interpolated_inputs: Sequence[str],
        time_integrated_outputs: Sequence[str],
        start_year: float = 2025.0,
        end_year: float = 2050.0,
        time_step: float = 1.0,
        interp_step: float = 5.0,
        cubic_interpolation: bool = False,
        differentiation_method: JAXDiscipline.DifferentiationMethod =
        JAXDiscipline.DifferentiationMethod.REVERSE,
        adjoint_method: AdjointMethod = AdjointMethod.RECURSIVE,
    ):
        self.models = list(models)
        self.constant_inputs = list(constant_inputs)
        self.control_delay_times = control_delay_times
        self.constrained_control_groups = constrained_control_groups
        self.custom_controls = custom_controls
        self.interpolated_inputs = list(interpolated_inputs)
        self.time_integrated_outputs = list(time_integrated_outputs)
        self.time_vector = append(
            arange(start_year, end_year, time_step), end_year
        )
        self.time_interpolation = append(
            arange(start_year, end_year, interp_step), end_year
        )
        self.cubic_interpolation = cubic_interpolation

        # self.time_integrated_outputs.extend([
        #     f"{group_name}.controls_constraint"
        #     for group_name in self.constrained_control_groups.keys()
        # ])

        if differentiation_method == JAXDiscipline.DifferentiationMethod.AUTO:
            raise ValueError("Chose either forward or reverse AutoDiff method.")
        if (adjoint_method in [
            self.AdjointMethod.RECURSIVE, self.AdjointMethod.BACKSOLVE
        ]) and differentiation_method == JAXDiscipline.DifferentiationMethod.FORWARD:
            raise ValueError(
                f"Forward AutoDiff is not compatible with {adjoint_method} adjoint."
            )

        if adjoint_method == self.AdjointMethod.DIRECT:
            self.__adjoint = DirectAdjoint()
        elif adjoint_method == self.AdjointMethod.RECURSIVE:
            self.__adjoint = RecursiveCheckpointAdjoint()
        else:
            self.__adjoint = BacksolveAdjoint()

        vectorized = []
        unvectorized = []
        for model in models:
            if all([
                var_name in self.constant_inputs or any([
                    var_name in disc.output_grammar.names
                    for disc in unvectorized
                ])
                for var_name in model.discipline.input_grammar.names
            ]):
                unvectorized.append(model.discipline)
            else:
                vectorized.append(model.discipline)

        self.unvectorized_chain = JAXChain(unvectorized)
        self.vectorized_chain = JAXChain(vectorized)

        self.non_modified_inputs = [
            name for name in self.vectorized_chain.input_grammar.names
            if name not in set(self.constant_inputs).union(
                self.interpolated_inputs
            ).union(self.control_delay_times.keys()).union(
                self.unvectorized_chain.output_grammar.names
            )
        ]

        default_inputs = {
            f"{self._constant_prefix}.{name}":
                self.unvectorized_chain.default_inputs[name]
            for name in self.unvectorized_chain.input_grammar.names
        }
        default_inputs.update({
            f"{self._constant_prefix}.{name}":
                self.vectorized_chain.default_inputs[name]
            for name in self.constant_inputs
            if name not in self.unvectorized_chain.input_grammar.names
        })
        default_inputs.update({
            f"{self._interpolation_prefix}.{name}":
                self.vectorized_chain.default_inputs[name] * ones(
                    self.time_interpolation.shape
                )
            for name in self.interpolated_inputs
        })
        default_inputs.update({
            f"{self._control_prefix}.{name}":
                self.vectorized_chain.default_inputs[name] * ones(
                    self.time_interpolation.shape
                )
            for name in self.control_delay_times.keys()
            if (
                    f"{self._control_prefix}.{name}" not in
                    self.unvectorized_chain.output_grammar.names
                ) and name not in self.custom_controls
        })
        default_inputs.update({
            name: self.vectorized_chain.default_inputs[name] * ones(
                self.time_vector.shape
            ) for name in self.non_modified_inputs
        })

        output_names = [
            f"{group_name}.controls_constraint"
            for group_name in self.constrained_control_groups.keys()
        ]
        output_names.extend([
            f"{group_name}.controls_constraint_violation"
            for group_name in self.constrained_control_groups.keys()
        ])
        output_names.extend(list(self.vectorized_chain.output_grammar.names))
        output_names.extend(list(self.unvectorized_chain.output_grammar.names))
        output_names.extend(self.constant_inputs)
        output_names.extend(self.interpolated_inputs)
        output_names.extend(self.control_delay_times.keys())
        output_names.extend([
            f"{self._interpolation_prefix}.{self._control_prefix}.{name}"
            for name in self.control_delay_times.keys()
        ])
        output_names.extend(
            [
                f"{self._integration_prefix}.{name}" for name in
                self.time_integrated_outputs
            ]
        )

        discipline = JAXDiscipline(
            name=name,
            function=self.__temporal_scenario,
            input_names=list(default_inputs.keys()),
            output_names=output_names,
            default_inputs=default_inputs,
            differentiation_method=differentiation_method,
        )
        discipline.set_cache_policy()
        super().__init__(discipline)

    def __temporal_scenario(self, input_data: DataType) -> DataType:
        output_data = {}

        # Run the unvectorized models
        unvectorized_inputs = {
            name: input_data[f"{self._constant_prefix}.{name}"]
            for name in self.unvectorized_chain.input_grammar.names
        }
        unvectorized_outputs = self.unvectorized_chain.jax_out_func(unvectorized_inputs)
        output_data.update(unvectorized_outputs)

        stacked_inputs = {
            name: value * ones(self.time_vector.shape)
            for name, value in unvectorized_outputs.items()
            if name in self.vectorized_chain.input_grammar.names
        }
        stacked_inputs.update({
            name: input_data[name] for name in self.non_modified_inputs
        })

        # Fixed inputs in time
        constant_inputs = {
            name: input_data[f"{self._constant_prefix}.{name}"] * ones(
                self.time_vector.shape
            )
            for name in self.constant_inputs
        }
        stacked_inputs.update(constant_inputs)
        output_data.update(constant_inputs)

        # Interpolated inputs in time
        interpolated_inputs = {
            name: interpolate_data(
                x=self.time_vector,
                x_data=self.time_interpolation,
                y_data=input_data[f"{self._interpolation_prefix}.{name}"],
                cubic=self.cubic_interpolation,
            )
            for name in self.interpolated_inputs
        }
        interpolated_inputs.update({
            f"{self._interpolation_prefix}.{self._control_prefix}.{name}":
                interpolate_data(
                    x=self.time_vector,
                    x_data=self.time_interpolation,
                    y_data=input_data[f"{self._control_prefix}.{name}"],
                    cubic=self.cubic_interpolation,
                )
            for name in self.control_delay_times.keys()
            if name not in self.custom_controls
        })
        interpolated_inputs.update(
            {
                f"{self._interpolation_prefix}.{self._control_prefix}.{name}":
                    interpolate_data(
                        x=self.time_vector,
                        x_data=output_data[f"{self._control_prefix}.{name}.times"],
                        y_data=output_data[f"{self._control_prefix}.{name}.values"],
                        cubic=self.cubic_interpolation,
                    )
                for name in self.custom_controls
            }
        )

        sum_controls = {
            group_name: sum(
                array([
                    interpolate_data(
                        x=self.time_vector,
                        x_data=output_data[f"{self._control_prefix}.{name}.times"]
                        if name in self.custom_controls else self.time_interpolation,
                        y_data=output_data[f"{self._control_prefix}.{name}.values"]
                        if name in self.custom_controls else input_data[
                            f"{self._control_prefix}.{name}"
                        ]
                    )
                for name in names
                ]), axis=0,
            ) for group_name, names in self.constrained_control_groups.items()
        }
        controls_constraint = {
            f"{group_name}.controls_constraint": values - 1.0
            for group_name, values in sum_controls.items()
        }
        controls_constraint.update({
            f"{group_name}.controls_constraint_violation":
                where(values > 1.0, values - 1.0, 0.0)
            for group_name, values in sum_controls.items()
        })
        output_data.update(controls_constraint)

        stacked_inputs.update(interpolated_inputs)
        output_data.update(interpolated_inputs)

        # Time-delayed inputs
        delayed_inputs = {
            name: diffeqsolve(
                terms=ODETerm(delay1_rhs),
                solver=Dopri5(),
                adjoint=self.__adjoint,
                t0=self.time_vector[0],
                t1=self.time_vector[-1],
                dt0=self.time_vector[1] - self.time_vector[0],
                saveat=SaveAt(ts=self.time_vector),
                y0=input_data[f"{self._control_prefix}.{name}"][0],
                args=(
                    delay_time,
                    input_data[f"{self._control_prefix}.{name}"],
                    self.time_interpolation,
                ),
            ).ys
            for name, delay_time in self.control_delay_times.items()
            if name not in self.custom_controls
        }
        delayed_inputs.update({
            name: diffeqsolve(
                terms=ODETerm(delay1_rhs),
                solver=Dopri5(),
                adjoint=self.__adjoint,
                t0=self.time_vector[0],
                t1=self.time_vector[-1],
                dt0=self.time_vector[1] - self.time_vector[0],
                saveat=SaveAt(ts=self.time_vector),
                y0=output_data[f"{self._control_prefix}.{name}.values"][0],
                args=(
                    self.control_delay_times[name],
                    output_data[f"{self._control_prefix}.{name}.values"],
                    output_data[f"{self._control_prefix}.{name}.times"],
                ),
            ).ys
            for name in self.custom_controls
        })
        stacked_inputs.update(delayed_inputs)
        output_data.update(delayed_inputs)

        # Vectorize time-step and compute all
        vectorized_func = vmap(self.vectorized_chain.jax_out_func)
        stacked_outputs = vectorized_func(stacked_inputs)
        output_data.update(stacked_outputs)

        time_integrated = {
            f"{self._integration_prefix}.{name}":
                trapezoid(output_data[name], self.time_vector)
            for name in self.time_integrated_outputs
            if name not in self.unvectorized_chain.output_grammar.names
        }
        output_data.update(time_integrated)
        return output_data

    @property
    def interpolation_prefix(self):
        return self._interpolation_prefix
