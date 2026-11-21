"""Deterministic measurement transforms for the prespecified training study."""

from math import pi

from sklearn.preprocessing import FunctionTransformer

from ai4i_pm.model_search import tree_pipeline
from ai4i_pm.models import check_model_inputs
from ai4i_pm.schema import PREDICTORS

TEMPERATURE = "temperature_difference_K"
POWER = "mechanical_power_W"


def temperature_difference(process_temperature_K, air_temperature_K):
    """Difference between process and ambient temperature, in kelvins."""
    return process_temperature_K - air_temperature_K


def mechanical_power(torque_Nm, rotational_speed_rpm):
    """Mechanical power in watts from torque and revolutions per minute."""
    return torque_Nm * (rotational_speed_rpm * (2 * pi / 60))


def engineer_features(frame, columns):
    check_model_inputs(frame)
    allowed = (*PREDICTORS, TEMPERATURE, POWER)
    if not columns or tuple(col for col in allowed if col in columns) != tuple(columns):
        raise ValueError("Expected unique authorized features in the predefined order")
    result = frame.loc[:, [col for col in columns if col in PREDICTORS]].copy()
    if TEMPERATURE in columns:
        result[TEMPERATURE] = temperature_difference(
            frame["Process temperature [K]"], frame["Air temperature [K]"]
        )
    if POWER in columns:
        result[POWER] = mechanical_power(frame["Torque [Nm]"], frame["Rotational speed [rpm]"])
    return result


def ablation_pipeline(estimator, columns):
    transformer = FunctionTransformer(engineer_features, kw_args={"columns": tuple(columns)})
    return tree_pipeline(estimator, [col for col in columns if col != "Type"], transformer)
