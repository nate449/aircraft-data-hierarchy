"""Behavior-verifying tests for systems_diagrams.py.

Each test asserts on the *output* of the functions under test —
diagram source content, formatted table substrings, or mock call arguments —
rather than merely checking that no exception was raised.
"""

import pytest
from unittest.mock import patch, MagicMock

import graphviz

# Import the diagram module directly so we can patch attributes on the actual
# module object. Patching via dotted string paths is unreliable here because
# work_breakdown_structure/__init__.py does `from .X import *` for several
# submodules, which causes mock's dotted-path resolver to occasionally land on
# the wrong namespace and raise AttributeError. patch.object(_diag_mod, "name")
# sidesteps the resolver entirely.
from aircraft_data_hierarchy.work_breakdown_structure.systems import (
    systems_diagrams as _diag_mod,
)

from aircraft_data_hierarchy.work_breakdown_structure.systems.systems_parameters import (
    System as ParameterSystem,
    SystemAttributes,
    FunctionalBlock,
    DataSignal,
    PhysicalCharacteristics,
    CoolingRequirements,
    PowerRequirements,
    FluidFlowCharacteristics,
    SignalType,
    SignalDirection,
)
from aircraft_data_hierarchy.work_breakdown_structure.systems.systems_diagrams import (
    create_system_diagram,
    create_system_attribute_tables,
    display_system_info,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_block_system():
    """Minimal system with two functional blocks connected by one data signal."""
    return ParameterSystem(
        wbs_id="1.2.3",
        mil_std_881f_reference="1.2.3",
        name="Hydraulic System",
        attributes=SystemAttributes(
            functional_blocks=[
                FunctionalBlock(
                    block_id="BLK_A",
                    name="Pump Controller",
                    description="Controls hydraulic pump speed",
                    inputs=["pressure_cmd"],
                    outputs=["motor_speed"],
                ),
                FunctionalBlock(
                    block_id="BLK_B",
                    name="Pressure Sensor",
                    description="Measures hydraulic pressure",
                    inputs=["line_pressure"],
                    outputs=["pressure_reading"],
                ),
            ],
            data_signals=[
                DataSignal(
                    name="PressureFeedback",
                    type=SignalType.ANALOG,
                    direction=SignalDirection.INPUT,
                    source="BLK_B",
                    destination="BLK_A",
                    description="Feedback signal from sensor to controller",
                ),
            ],
            physical_characteristics=PhysicalCharacteristics(
                weight=10.5,
                dimensions={"length": 0.5, "width": 0.3, "height": 0.2},
                volume=0.03,
                center_of_gravity={"x": 1.0, "y": 0.0, "z": 0.5},
            ),
            cooling_requirements=CoolingRequirements(
                method="Air",
                heat_dissipation=150.0,
                max_operating_temperature=85.0,
            ),
            power_requirements=PowerRequirements(
                voltage=28.0,
                current=5.25,
                frequency=400.0,
                power_type="AC",
                peak_power=200.0,
                average_power=147.0,
            ),
        ),
        components=["1.2.3.1"],
    )


@pytest.fixture
def system_with_fluid(two_block_system):
    """Same system but with fluid-flow characteristics populated."""
    data = two_block_system.model_dump()
    data["attributes"]["fluid_flow"] = {
        "fluid_type": "Hydraulic Oil",
        "flow_rate": 12.5,
        "max_pressure": 21000000.0,
        "min_pressure": 500000.0,
        "temperature_range": (-40.0, 120.0),
        "viscosity": 0.000032,
        "density": 860.0,
    }
    return ParameterSystem(**data)


@pytest.fixture
def system_no_frequency(two_block_system):
    """System whose power_requirements.frequency is None."""
    data = two_block_system.model_dump()
    data["attributes"]["power_requirements"]["frequency"] = None
    return ParameterSystem(**data)


# ---------------------------------------------------------------------------
# create_system_diagram
# ---------------------------------------------------------------------------

def test_diagram_contains_node_names(two_block_system):
    dot = create_system_diagram(two_block_system)
    assert "Pump Controller" in dot.source
    assert "Pressure Sensor" in dot.source


def test_diagram_contains_node_ids(two_block_system):
    dot = create_system_diagram(two_block_system)
    assert "BLK_A" in dot.source
    assert "BLK_B" in dot.source


def test_diagram_edge_connects_source_to_destination(two_block_system):
    dot = create_system_diagram(two_block_system)
    assert "BLK_B -> BLK_A" in dot.source


def test_diagram_edge_label(two_block_system):
    dot = create_system_diagram(two_block_system)
    assert "PressureFeedback" in dot.source


def test_diagram_comment_contains_system_name(two_block_system):
    dot = create_system_diagram(two_block_system)
    assert "Hydraulic System" in dot.comment


def test_diagram_rankdir_is_left_to_right(two_block_system):
    dot = create_system_diagram(two_block_system)
    assert "rankdir=LR" in dot.source


def test_diagram_returns_digraph(two_block_system):
    dot = create_system_diagram(two_block_system)
    assert isinstance(dot, graphviz.Digraph)


def test_diagram_multiple_signals():
    """Two signals between three blocks produce two edges."""
    system = ParameterSystem(
        wbs_id="1.2.4",
        mil_std_881f_reference="1.2.4",
        name="ECS",
        attributes=SystemAttributes(
            functional_blocks=[
                FunctionalBlock(block_id="B1", name="Valve", description="d", inputs=[], outputs=[]),
                FunctionalBlock(block_id="B2", name="Actuator", description="d", inputs=[], outputs=[]),
                FunctionalBlock(block_id="B3", name="Controller", description="d", inputs=[], outputs=[]),
            ],
            data_signals=[
                DataSignal(name="Cmd", type=SignalType.DIGITAL, direction=SignalDirection.OUTPUT,
                           source="B3", destination="B2", description="d"),
                DataSignal(name="Status", type=SignalType.ANALOG, direction=SignalDirection.INPUT,
                           source="B1", destination="B3", description="d"),
            ],
            physical_characteristics=PhysicalCharacteristics(
                weight=5.0, dimensions={"length": 0.1, "width": 0.1, "height": 0.1},
                volume=0.001, center_of_gravity={"x": 0.0, "y": 0.0, "z": 0.0},
            ),
            cooling_requirements=CoolingRequirements(method="Liquid", heat_dissipation=50.0, max_operating_temperature=70.0),
            power_requirements=PowerRequirements(voltage=115.0, current=2.0, power_type="AC", peak_power=300.0, average_power=230.0),
        ),
        components=["1.2.4.1"],
    )
    dot = create_system_diagram(system)
    assert "B3 -> B2" in dot.source
    assert "B1 -> B3" in dot.source
    assert "Cmd" in dot.source
    assert "Status" in dot.source


# ---------------------------------------------------------------------------
# create_system_attribute_tables
# ---------------------------------------------------------------------------

def test_tables_count_without_fluid(two_block_system):
    tables = create_system_attribute_tables(two_block_system)
    assert len(tables) == 3


def test_tables_count_with_fluid(system_with_fluid):
    tables = create_system_attribute_tables(system_with_fluid)
    assert len(tables) == 4


def test_physical_characteristics_title_and_values(two_block_system):
    tables = create_system_attribute_tables(two_block_system)
    title, html = tables[0]
    assert title == "Physical Characteristics"
    # tabulate right-aligns numeric columns and strips trailing zeros
    assert "Weight" in html
    assert "10.5" in html
    assert "Length" in html
    assert "Width" in html
    assert "Height" in html
    assert "Volume" in html
    assert "0.03" in html          # volume
    assert "Center of Gravity X" in html
    assert "Center of Gravity Y" in html
    assert "Center of Gravity Z" in html
    assert "kg" in html
    assert "m³" in html


def test_cooling_requirements_values(two_block_system):
    tables = create_system_attribute_tables(two_block_system)
    title, html = tables[1]
    assert title == "Cooling Requirements"
    assert "Air" in html
    assert "150.0" in html
    assert "85.0" in html


def test_power_requirements_with_frequency(two_block_system):
    tables = create_system_attribute_tables(two_block_system)
    title, html = tables[2]
    assert title == "Power Requirements"
    assert "28.0" in html
    assert "5.25" in html
    assert "400.0" in html
    assert "AC" in html
    assert "200.0" in html
    assert "147.0" in html


def test_power_requirements_frequency_none_shows_na(system_no_frequency):
    tables = create_system_attribute_tables(system_no_frequency)
    _, html = tables[2]
    assert "N/A" in html


def test_fluid_flow_values(system_with_fluid):
    tables = create_system_attribute_tables(system_with_fluid)
    title, html = tables[3]
    assert title == "Fluid Flow Characteristics"
    assert "Hydraulic Oil" in html
    assert "12.50" in html       # flow_rate
    assert "21000.00" in html    # max_pressure / 1e3
    assert "500.00" in html      # min_pressure / 1e3
    assert "-40.0" in html       # temp min
    assert "120.0" in html       # temp max
    assert "0.000032" in html    # viscosity
    assert "860.0" in html       # density
    assert "L/min" in html
    assert "kPa" in html


def test_fluid_flow_absent_when_none(two_block_system):
    tables = create_system_attribute_tables(two_block_system)
    titles = [t for t, _ in tables]
    assert "Fluid Flow Characteristics" not in titles


def test_tables_are_html_formatted(two_block_system):
    tables = create_system_attribute_tables(two_block_system)
    for _, html in tables:
        assert "<table" in html.lower()


# ---------------------------------------------------------------------------
# display_system_info
# ---------------------------------------------------------------------------

_DIAG_MODULE = "aircraft_data_hierarchy.work_breakdown_structure.systems.systems_diagrams"


@patch.object(_diag_mod, "display")
@patch.object(_diag_mod, "Image")
@patch.object(_diag_mod, "HTML")
@patch("graphviz.Digraph.render")
def test_display_renders_png(mock_render, mock_html, mock_image, mock_display, two_block_system):
    display_system_info(two_block_system)
    mock_render.assert_called_once_with("system_diagram", format="png", cleanup=True)


@patch.object(_diag_mod, "display")
@patch.object(_diag_mod, "Image")
@patch.object(_diag_mod, "HTML")
@patch("graphviz.Digraph.render")
def test_display_creates_image_with_correct_path(mock_render, mock_html, mock_image, mock_display, two_block_system):
    display_system_info(two_block_system)
    mock_image.assert_called_once_with("system_diagram.png")


@patch.object(_diag_mod, "display")
@patch.object(_diag_mod, "Image")
@patch.object(_diag_mod, "HTML")
@patch("graphviz.Digraph.render")
def test_display_calls_display_twice(mock_render, mock_html, mock_image, mock_display, two_block_system):
    display_system_info(two_block_system)
    assert mock_display.call_count == 2


@patch.object(_diag_mod, "display")
@patch.object(_diag_mod, "Image")
@patch.object(_diag_mod, "HTML")
@patch("graphviz.Digraph.render")
def test_display_first_call_is_image(mock_render, mock_html, mock_image, mock_display, two_block_system):
    display_system_info(two_block_system)
    first_call_arg = mock_display.call_args_list[0][0][0]
    assert first_call_arg is mock_image.return_value


@patch.object(_diag_mod, "display")
@patch.object(_diag_mod, "Image")
@patch.object(_diag_mod, "HTML")
@patch("graphviz.Digraph.render")
def test_display_second_call_is_html(mock_render, mock_html, mock_image, mock_display, two_block_system):
    display_system_info(two_block_system)
    second_call_arg = mock_display.call_args_list[1][0][0]
    assert second_call_arg is mock_html.return_value


@patch.object(_diag_mod, "display")
@patch.object(_diag_mod, "Image")
@patch.object(_diag_mod, "HTML")
@patch("graphviz.Digraph.render")
def test_display_html_contains_table_titles(mock_render, mock_html, mock_image, mock_display, two_block_system):
    display_system_info(two_block_system)
    html_arg = mock_html.call_args[0][0]
    assert "Physical Characteristics" in html_arg
    assert "Cooling Requirements" in html_arg
    assert "Power Requirements" in html_arg
