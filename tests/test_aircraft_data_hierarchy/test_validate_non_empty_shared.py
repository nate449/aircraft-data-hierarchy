"""Cross-model parametrised tests for the duplicated validate_non_empty validators.

Exercises whitespace-rejection and strip-on-acceptance logic shared by:
  - Requirement  (requirements.py)
  - Requirements (requirements.py)
  - Activity     (behavior.py)
  - Behavior     (behavior.py)
  - String       (airframe_geometry.py)
"""

import pytest
from pydantic import ValidationError

from aircraft_data_hierarchy.requirements import Requirement, Requirements
from aircraft_data_hierarchy.behavior import Activity, Behavior
from aircraft_data_hierarchy.work_breakdown_structure.airframe.airframe_geometry import String


# ---------------------------------------------------------------------------
# Helpers – factory kwargs for constructing valid instances
# ---------------------------------------------------------------------------

REQUIREMENT_DEFAULTS = {
    "name": "REQ-001",
    "description": "Valid description",
    "priority": "high",
    "verification_method": "test",
    "status": "open",
    "acceptance_criteria": "Must pass.",
}

REQUIREMENTS_DEFAULTS = {
    "name": "Req Set",
    "description": "Valid set description",
}

STRING_DEFAULTS = {
    "value": "valid",
}


# ---------------------------------------------------------------------------
# Whitespace inputs that must be rejected
# ---------------------------------------------------------------------------

WHITESPACE_INPUTS = [
    pytest.param("", id="empty"),
    pytest.param(" ", id="single-space"),
    pytest.param("   ", id="multiple-spaces"),
    pytest.param("\t", id="tab"),
    pytest.param("\n", id="newline"),
    pytest.param("\r\n", id="crlf"),
    pytest.param(" \t\n\r ", id="mixed-whitespace"),
]


# ---------------------------------------------------------------------------
# 1. Requirement – 6 validated fields × 7 whitespace inputs
# ---------------------------------------------------------------------------

REQUIREMENT_VALIDATED_FIELDS = [
    "name",
    "description",
    "priority",
    "verification_method",
    "status",
    "acceptance_criteria",
]


@pytest.mark.parametrize("field", REQUIREMENT_VALIDATED_FIELDS)
@pytest.mark.parametrize("bad_value", WHITESPACE_INPUTS)
def test_requirement_rejects_whitespace(field, bad_value):
    kwargs = {**REQUIREMENT_DEFAULTS, field: bad_value}
    with pytest.raises(ValidationError):
        Requirement(**kwargs)


@pytest.mark.parametrize("field", REQUIREMENT_VALIDATED_FIELDS)
def test_requirement_accepts_and_strips(field):
    kwargs = {**REQUIREMENT_DEFAULTS, field: "  valid  "}
    req = Requirement(**kwargs)
    assert getattr(req, field) == "valid"


@pytest.mark.parametrize("field", REQUIREMENT_VALIDATED_FIELDS)
def test_requirement_accepts_minimal(field):
    kwargs = {**REQUIREMENT_DEFAULTS, field: "a"}
    req = Requirement(**kwargs)
    assert getattr(req, field) == "a"


@pytest.mark.parametrize("field", REQUIREMENT_VALIDATED_FIELDS)
def test_requirement_strips_tabs_newlines(field):
    kwargs = {**REQUIREMENT_DEFAULTS, field: "\thello\n"}
    req = Requirement(**kwargs)
    assert getattr(req, field) == "hello"


# ---------------------------------------------------------------------------
# 2. Requirements – name and description
# ---------------------------------------------------------------------------

REQUIREMENTS_VALIDATED_FIELDS = ["name", "description"]


@pytest.mark.parametrize("field", REQUIREMENTS_VALIDATED_FIELDS)
@pytest.mark.parametrize("bad_value", WHITESPACE_INPUTS)
def test_requirements_rejects_whitespace(field, bad_value):
    kwargs = {**REQUIREMENTS_DEFAULTS, field: bad_value}
    with pytest.raises(ValidationError):
        Requirements(**kwargs)


@pytest.mark.parametrize("field", REQUIREMENTS_VALIDATED_FIELDS)
def test_requirements_accepts_and_strips(field):
    kwargs = {**REQUIREMENTS_DEFAULTS, field: "  valid  "}
    req = Requirements(**kwargs)
    assert getattr(req, field) == "valid"


@pytest.mark.parametrize("field", REQUIREMENTS_VALIDATED_FIELDS)
def test_requirements_strips_tabs_newlines(field):
    kwargs = {**REQUIREMENTS_DEFAULTS, field: "\thello\n"}
    req = Requirements(**kwargs)
    assert getattr(req, field) == "hello"


# ---------------------------------------------------------------------------
# 3. Activity – name and description (Optional[str])
# ---------------------------------------------------------------------------

ACTIVITY_VALIDATED_FIELDS = ["name", "description"]


@pytest.mark.parametrize("field", ACTIVITY_VALIDATED_FIELDS)
@pytest.mark.parametrize("bad_value", WHITESPACE_INPUTS)
def test_activity_rejects_whitespace(field, bad_value):
    with pytest.raises(ValidationError):
        Activity(**{field: bad_value})


@pytest.mark.parametrize("field", ACTIVITY_VALIDATED_FIELDS)
def test_activity_accepts_none(field):
    a = Activity(**{field: None})
    assert getattr(a, field) is None


def test_activity_default_construction():
    a = Activity()
    assert a.name is None
    assert a.description is None


@pytest.mark.parametrize("field", ACTIVITY_VALIDATED_FIELDS)
def test_activity_accepts_padded_string(field):
    # NOTE: Activity's validate_non_empty does not strip; CommonBaseModel.strip_strings
    # does not apply because the subclass validator overrides the wildcard for these fields.
    a = Activity(**{field: "  valid  "})
    assert getattr(a, field) == "  valid  "


@pytest.mark.parametrize("field", ACTIVITY_VALIDATED_FIELDS)
def test_activity_preserves_tabs_newlines(field):
    # NOTE: stripping is not applied — see test_activity_accepts_padded_string.
    a = Activity(**{field: "\thello\n"})
    assert getattr(a, field) == "\thello\n"


@pytest.mark.parametrize("field", ACTIVITY_VALIDATED_FIELDS)
def test_activity_accepts_minimal(field):
    a = Activity(**{field: "a"})
    assert getattr(a, field) == "a"


# ---------------------------------------------------------------------------
# 4. Behavior – name and description (Optional[str], mode="before")
# ---------------------------------------------------------------------------

BEHAVIOR_VALIDATED_FIELDS = ["name", "description"]


@pytest.mark.parametrize("field", BEHAVIOR_VALIDATED_FIELDS)
@pytest.mark.parametrize("bad_value", WHITESPACE_INPUTS)
def test_behavior_rejects_whitespace(field, bad_value):
    with pytest.raises(ValidationError):
        Behavior(**{field: bad_value})


@pytest.mark.parametrize("field", BEHAVIOR_VALIDATED_FIELDS)
def test_behavior_accepts_none(field):
    b = Behavior(**{field: None})
    assert getattr(b, field) is None


def test_behavior_default_construction():
    b = Behavior()
    assert b.name is None
    assert b.description is None


@pytest.mark.parametrize("field", BEHAVIOR_VALIDATED_FIELDS)
def test_behavior_accepts_padded_string(field):
    # NOTE: Behavior's validate_non_empty (mode="before") does not strip;
    # CommonBaseModel.strip_strings does not apply because the subclass validator
    # overrides the wildcard for these fields.
    b = Behavior(**{field: "  valid  "})
    assert getattr(b, field) == "  valid  "


@pytest.mark.parametrize("field", BEHAVIOR_VALIDATED_FIELDS)
def test_behavior_preserves_tabs_newlines(field):
    # NOTE: stripping is not applied — see test_behavior_accepts_padded_string.
    b = Behavior(**{field: "\thello\n"})
    assert getattr(b, field) == "\thello\n"


@pytest.mark.parametrize("field", BEHAVIOR_VALIDATED_FIELDS)
def test_behavior_accepts_minimal(field):
    b = Behavior(**{field: "a"})
    assert getattr(b, field) == "a"


# ---------------------------------------------------------------------------
# 5. String – value (required str, mode="before")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_value", WHITESPACE_INPUTS)
def test_string_rejects_whitespace(bad_value):
    with pytest.raises(ValidationError):
        String(value=bad_value)


def test_string_accepts_and_strips():
    s = String(value="  x  ")
    assert s.value == "x"


def test_string_strips_tabs_newlines():
    s = String(value="\thello\n")
    assert s.value == "hello"


def test_string_accepts_minimal():
    s = String(value="a")
    assert s.value == "a"
