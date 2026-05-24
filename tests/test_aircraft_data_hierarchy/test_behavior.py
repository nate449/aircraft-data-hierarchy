"""Tests for behavior.py — covering validators on uncovered branches (lines 266-268, 513-516, 636-647, 955-957, 962-966, 1006, 1022-1024)."""
import pytest
from pydantic import ValidationError

from aircraft_data_hierarchy.behavior import (
    Uncertainty, UncertaintyEffect, NormalPDF, UniformPDF, Bounds,
    Function, IndependentVarPts, DependentVarPts, IndependentVarRef, DependentVarRef, FunctionDefn,
    GriddedTableRef, GriddedTableDef, GriddedTable, UngriddedTableRef,
    Activity, ActivityState, Behavior,
)


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty.validate_pdf (lines 266-268)
# Only one of normal_pdf or uniform_pdf can be specified.
# ──────────────────────────────────────────────────────────────────────────────

class TestUncertaintyPDFValidator:
    def test_normal_pdf_alone_accepted(self):
        u = Uncertainty(
            effect=UncertaintyEffect.ADDITIVE,
            normal_pdf=NormalPDF(num_sigmas=2.0),
        )
        assert u.normal_pdf is not None
        assert u.uniform_pdf is None

    def test_uniform_pdf_alone_accepted(self):
        u = Uncertainty(
            effect=UncertaintyEffect.MULTIPLICATIVE,
            uniform_pdf=UniformPDF(bounds=[Bounds(value="0.5")]),
        )
        assert u.uniform_pdf is not None
        assert u.normal_pdf is None

    def test_both_pdfs_not_caught_by_field_validator(self):
        """Demonstrates that the validate_pdf field_validator (lines 266-268) is unreachable:
        info.data never contains the field currently being validated in Pydantic v2,
        so the 'both specified' branch can never fire. This is a latent bug."""
        u = Uncertainty(
            effect=UncertaintyEffect.ADDITIVE,
            normal_pdf=NormalPDF(num_sigmas=2.0),
            uniform_pdf=UniformPDF(bounds=[Bounds(value="0.5")]),
        )
        # Both PDFs coexist because the validator logic is unreachable
        assert u.normal_pdf is not None
        assert u.uniform_pdf is not None


# ──────────────────────────────────────────────────────────────────────────────
# Function.validate_function_type (lines 513-516)
# Can't have both simple (independent_var_pts) and complex (independent_var_ref) representations.
# ──────────────────────────────────────────────────────────────────────────────

class TestFunctionValidator:
    def test_simple_representation_accepted(self):
        f = Function(
            name="simple_func",
            independent_var_pts=[IndependentVarPts(var_id="x", value="1 2 3")],
            dependent_var_pts=DependentVarPts(var_id="y", value="4 5 6"),
        )
        assert f.independent_var_pts is not None
        assert f.dependent_var_pts is not None

    def test_complex_representation_accepted(self):
        f = Function(
            name="complex_func",
            independent_var_ref=[IndependentVarRef(var_id="x", min=0.0, max=10.0)],
            dependent_var_ref=[DependentVarRef(var_id="y")],
            function_defn=FunctionDefn(name="lookup", gridded_table_ref=GriddedTableRef(gt_id="gt1")),
        )
        assert f.independent_var_ref is not None
        assert f.function_defn is not None

    def test_both_simple_and_complex_rejected_via_dependent_var_ref(self):
        """dependent_var_ref is declared after independent_var_ref, so the validator fires."""
        with pytest.raises(ValidationError, match="both simple and complex"):
            Function(
                name="invalid_func",
                independent_var_pts=[IndependentVarPts(var_id="x", value="1 2 3")],
                independent_var_ref=[IndependentVarRef(var_id="x", min=0.0, max=10.0)],
                dependent_var_ref=[DependentVarRef(var_id="y")],
            )

    def test_both_simple_and_complex_rejected_via_function_defn(self):
        with pytest.raises(ValidationError, match="both simple and complex"):
            Function(
                name="invalid_func",
                independent_var_pts=[IndependentVarPts(var_id="x", value="1 2 3")],
                independent_var_ref=[IndependentVarRef(var_id="x", min=0.0, max=10.0)],
                function_defn=FunctionDefn(name="f", gridded_table_ref=GriddedTableRef(gt_id="gt1")),
            )


# ──────────────────────────────────────────────────────────────────────────────
# FunctionDefn.validate_table_type (lines 636-647)
# Only one table type can be specified in a function definition.
# ──────────────────────────────────────────────────────────────────────────────

class TestFunctionDefnValidator:
    def test_single_table_type_accepted(self):
        fd = FunctionDefn(name="lookup", gridded_table_ref=GriddedTableRef(gt_id="gt1"))
        assert fd.gridded_table_ref is not None

    def test_three_table_types_rejected(self):
        """With 3 table fields set, the third field's validator sees 2 already in info.data → fires."""
        with pytest.raises(ValidationError, match="Only one table type"):
            FunctionDefn(
                name="invalid",
                gridded_table_ref=GriddedTableRef(gt_id="gt1"),
                gridded_table_def=GriddedTableDef(name="def1"),
                ungridded_table_ref=UngriddedTableRef(ut_id="ut1"),
            )

    def test_two_table_types_not_caught(self):
        """With only 2 table fields, the second field's validator sees only 1 prior → sum=1, not >1.
        This documents the validator's limitation with only 2 table types."""
        fd = FunctionDefn(
            name="two_tables",
            gridded_table_ref=GriddedTableRef(gt_id="gt1"),
            ungridded_table_ref=UngriddedTableRef(ut_id="ut1"),
        )
        # Both coexist because the validator threshold is > 1 (not >= 1)
        assert fd.gridded_table_ref is not None
        assert fd.ungridded_table_ref is not None


# ──────────────────────────────────────────────────────────────────────────────
# Activity validators (lines 955-957, 962-966)
# ──────────────────────────────────────────────────────────────────────────────

class TestActivityValidators:
    def test_valid_activity(self):
        a = Activity(name="Design Review", description="Review design docs", state=ActivityState.PENDING)
        assert a.name == "Design Review"
        assert a.state == ActivityState.PENDING

    @pytest.mark.parametrize("field", ["name", "description"])
    def test_empty_string_rejected(self, field):
        with pytest.raises(ValidationError, match="must not be empty"):
            Activity(**{field: ""})

    @pytest.mark.parametrize("field", ["name", "description"])
    def test_whitespace_only_rejected(self, field):
        with pytest.raises(ValidationError, match="must not be empty"):
            Activity(**{field: "   "})

    def test_valid_dependencies(self):
        a = Activity(name="Build", dependencies=["Design", "Prototype"])
        assert a.dependencies == ["Design", "Prototype"]

    def test_empty_dependency_name_rejected(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            Activity(name="Build", dependencies=["Design", ""])

    def test_whitespace_dependency_name_rejected(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            Activity(name="Build", dependencies=["  "])


# ──────────────────────────────────────────────────────────────────────────────
# Behavior validators (lines 1006, 1022-1024)
# ──────────────────────────────────────────────────────────────────────────────

class TestBehaviorValidators:
    def test_valid_behavior(self):
        b = Behavior(
            name="Flight Test",
            description="Execute flight test sequence",
            sequence=[{"name": "Preflight"}],
        )
        assert b.name == "Flight Test"
        assert len(b.sequence) == 1

    @pytest.mark.parametrize("field", ["name", "description"])
    def test_empty_string_rejected(self, field):
        with pytest.raises(ValidationError, match="must not be empty"):
            Behavior(**{field: ""})

    @pytest.mark.parametrize("field", ["name", "description"])
    def test_whitespace_only_rejected(self, field):
        with pytest.raises(ValidationError, match="must not be empty"):
            Behavior(**{field: "   "})

    def test_empty_sequence_rejected(self):
        with pytest.raises(ValidationError, match="must contain at least one activity"):
            Behavior(name="Test", sequence=[])

    def test_none_sequence_accepted(self):
        b = Behavior(name="Test")
        assert b.sequence is None
