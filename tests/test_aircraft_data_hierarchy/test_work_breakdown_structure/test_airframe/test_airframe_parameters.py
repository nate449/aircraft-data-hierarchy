"""Tests for airframe_parameters models — focused on validator behavior, boundaries, and cross-field consistency."""
import pytest
from pydantic import ValidationError

from aircraft_data_hierarchy.work_breakdown_structure.airframe.airframe_parameters import (
    ReferenceData, FlightConditions, ConfigurationLayout, Airfoil,
    PlanformType, LiftingSurface, TwinVerticalTail,
    GroundEffectsDefinition, FlapType, NoseType, BlowingType, SymmetricFlap,
    ControlType, AsymmetricControl, BodyShape,
    TailShape, Body, LowAspectRatioWingBody, TransverseJetControl,
    HypersonicFlapControl, EngineType, PropellerPowerProperties, JetEngineType,
    JetPowerProperties, AerodynamicsData
)
from aircraft_data_hierarchy.work_breakdown_structure.airframe.airframe_geometry import Point, Spline


# ──────────────────────────────────────────────────────────────────────────────
# ReferenceData
# ──────────────────────────────────────────────────────────────────────────────

class TestReferenceData:
    def test_happy_path(self):
        model = ReferenceData(RougHgt=0.01, Sref=100.0, Cbar=10.0, BLref=5.0)
        assert model.roughness == pytest.approx(0.01, abs=1e-6)
        assert model.reference_area == pytest.approx(100.0, abs=1e-6)
        assert model.reference_length == pytest.approx(10.0, abs=1e-6)

    @pytest.mark.parametrize("roughness", [0.0, 0.02])
    def test_roughness_at_boundaries(self, roughness):
        model = ReferenceData(RougHgt=roughness, Sref=1.0, Cbar=1.0, BLref=1.0)
        assert model.roughness == pytest.approx(roughness, abs=1e-6)

    @pytest.mark.parametrize("roughness", [-0.01, 0.03, 0.5])
    def test_roughness_outside_boundaries(self, roughness):
        with pytest.raises(ValidationError, match="roughness|less than or equal|greater than or equal"):
            ReferenceData(RougHgt=roughness, Sref=1.0, Cbar=1.0, BLref=1.0)

    @pytest.mark.parametrize("field,alias", [
        ("reference_area", "Sref"),
        ("reference_length", "Cbar"),
        ("lateral_reference", "BLref"),
    ])
    def test_positive_fields_reject_zero(self, field, alias):
        with pytest.raises(ValidationError, match="greater than"):
            ReferenceData(**{alias: 0.0})

    @pytest.mark.parametrize("field,alias", [
        ("reference_area", "Sref"),
        ("reference_length", "Cbar"),
        ("lateral_reference", "BLref"),
    ])
    def test_positive_fields_reject_negative(self, field, alias):
        with pytest.raises(ValidationError, match="greater than"):
            ReferenceData(**{alias: -1.0})


# ──────────────────────────────────────────────────────────────────────────────
# FlightConditions
# ──────────────────────────────────────────────────────────────────────────────

class TestFlightConditions:
    def test_happy_path(self):
        model = FlightConditions(
            loop_control=2, qty_machs=2, machs=[0.5, 0.8],
            qty_alphas=2, alphas=[5.0, 10.0],
            qty_altitudes=2, altitudes=[10000.0, 20000.0],
        )
        assert model.loop_control == 2
        assert model.machs == [0.5, 0.8]
        assert model.altitudes == [10000.0, 20000.0]

    @pytest.mark.parametrize("value", [0, 4, -1])
    def test_loop_control_out_of_range(self, value):
        with pytest.raises(ValidationError, match="greater than|less than"):
            FlightConditions(loop_control=value)

    def test_negative_mach_rejected(self):
        with pytest.raises(ValidationError, match="non-negative"):
            FlightConditions(machs=[-0.5, 0.8])

    def test_negative_altitude_rejected(self):
        with pytest.raises(ValidationError, match="non-negative"):
            FlightConditions(altitudes=[-100.0])

    def test_list_length_mismatch_raises(self):
        with pytest.raises(ValidationError, match="must match"):
            FlightConditions(qty_machs=3, machs=[0.5, 0.8])

    def test_list_length_mismatch_alphas(self):
        with pytest.raises(ValidationError, match="must match"):
            FlightConditions(qty_alphas=1, alphas=[5.0, 10.0])

    def test_list_length_mismatch_altitudes(self):
        with pytest.raises(ValidationError, match="must match"):
            FlightConditions(qty_altitudes=5, altitudes=[10000.0])

    @pytest.mark.parametrize("mach", [0.6, 0.98])
    def test_transonic_mach_boundaries(self, mach):
        model = FlightConditions(transonic_mach=mach)
        assert model.transonic_mach == pytest.approx(mach, abs=1e-6)

    @pytest.mark.parametrize("mach", [0.59, 0.99, 1.0])
    def test_transonic_mach_out_of_range(self, mach):
        with pytest.raises(ValidationError, match="greater than or equal|less than"):
            FlightConditions(transonic_mach=mach)

    @pytest.mark.parametrize("mach", [1.01, 1.39])
    def test_supersonic_mach_boundaries(self, mach):
        model = FlightConditions(supersonic_mach=mach)
        assert model.supersonic_mach == pytest.approx(mach, abs=1e-6)

    @pytest.mark.parametrize("mach", [1.0, 1.4, 2.0])
    def test_supersonic_mach_out_of_range(self, mach):
        with pytest.raises(ValidationError, match="greater than or equal|less than"):
            FlightConditions(supersonic_mach=mach)

    def test_flight_path_angle_boundaries(self):
        model = FlightConditions(flight_path_angle=0.01)
        assert model.flight_path_angle == pytest.approx(0.01, abs=1e-6)
        model2 = FlightConditions(flight_path_angle=1.99)
        assert model2.flight_path_angle == pytest.approx(1.99, abs=1e-6)

    @pytest.mark.parametrize("angle", [0.0, -1.0, 2.0, 5.0])
    def test_flight_path_angle_out_of_range(self, angle):
        with pytest.raises(ValidationError, match="greater than|less than"):
            FlightConditions(flight_path_angle=angle)


# ──────────────────────────────────────────────────────────────────────────────
# ConfigurationLayout
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigurationLayout:
    def test_happy_path(self):
        model = ConfigurationLayout(
            center_of_gravity_station=10.0,
            wing_apex_station=25.0,
            model_scale=1.5,
        )
        assert model.center_of_gravity_station == pytest.approx(10.0, abs=1e-6)
        assert model.wing_apex_station == pytest.approx(25.0, abs=1e-6)
        assert model.model_scale == pytest.approx(1.5, abs=1e-6)

    @pytest.mark.parametrize("field", [
        "canard_apex_station", "wing_apex_station",
        "horizontal_apex_station", "vertical_apex_station", "fin_apex_station",
    ])
    def test_ge0_fields_reject_negative(self, field):
        with pytest.raises(ValidationError, match="non-negative|greater than or equal"):
            ConfigurationLayout(**{field: -1.0})

    def test_model_scale_must_be_positive(self):
        with pytest.raises(ValidationError, match="greater than"):
            ConfigurationLayout(model_scale=0.0)

    def test_model_scale_rejects_negative(self):
        with pytest.raises(ValidationError, match="non-negative|greater than"):
            ConfigurationLayout(model_scale=-1.0)


# ──────────────────────────────────────────────────────────────────────────────
# LiftingSurface
# ──────────────────────────────────────────────────────────────────────────────

class TestLiftingSurface:
    def test_happy_path(self):
        model = LiftingSurface(
            tip_chord=1.0, root_chord=6.0, reference_chord_fraction=0.25
        )
        assert model.tip_chord == pytest.approx(1.0, abs=1e-6)
        assert model.root_chord == pytest.approx(6.0, abs=1e-6)
        assert model.reference_chord_fraction == pytest.approx(0.25, abs=1e-6)

    @pytest.mark.parametrize("field", [
        "tip_chord", "outboard_panel_semi_span", "exposed_panel_semi_span",
        "total_panel_semi_span", "breakpoint_chord", "root_chord",
    ])
    def test_non_negative_fields_reject_negative(self, field):
        with pytest.raises(ValidationError, match="non-negative|must be non"):
            LiftingSurface(**{field: -0.1})

    @pytest.mark.parametrize("value", [0.0, -0.1, 1.0, 1.5])
    def test_reference_chord_fraction_must_be_in_0_1_exclusive(self, value):
        with pytest.raises(ValidationError, match="chord fraction|greater than|less than"):
            LiftingSurface(reference_chord_fraction=value)

    @pytest.mark.parametrize("value", [0.01, 0.5, 0.99])
    def test_reference_chord_fraction_valid(self, value):
        model = LiftingSurface(reference_chord_fraction=value)
        assert model.reference_chord_fraction == pytest.approx(value, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# TwinVerticalTail
# ──────────────────────────────────────────────────────────────────────────────

class TestTwinVerticalTail:
    def test_happy_path(self):
        model = TwinVerticalTail(span_above=1.5, total_span=3.0, body_depth=2.0)
        assert model.span_above == pytest.approx(1.5, abs=1e-6)
        assert model.total_span == pytest.approx(3.0, abs=1e-6)

    @pytest.mark.parametrize("field", [
        "span_above", "total_span", "body_depth", "separation",
        "planform_area", "closure_angle", "lateral_arm", "vertical_arm",
    ])
    def test_all_fields_reject_negative(self, field):
        with pytest.raises(ValidationError, match="not be negative|must not be"):
            TwinVerticalTail(**{field: -0.01})


# ──────────────────────────────────────────────────────────────────────────────
# GroundEffectsDefinition
# ──────────────────────────────────────────────────────────────────────────────

class TestGroundEffectsDefinition:
    def test_happy_path(self):
        model = GroundEffectsDefinition(qty_heights=3, heights=[100.0, 200.0, 300.0])
        assert model.qty_heights == 3
        assert len(model.heights) == 3

    def test_negative_height_rejected(self):
        with pytest.raises(ValidationError, match="must be between 0 and 1000"):
            GroundEffectsDefinition(heights=[-1.0, 100.0])

    def test_height_above_1000_rejected(self):
        with pytest.raises(ValidationError, match="must be between 0 and 1000"):
            GroundEffectsDefinition(heights=[1001.0])

    @pytest.mark.parametrize("heights", [[0.0], [500.0, 1000.0]])
    def test_boundary_heights_accepted(self, heights):
        model = GroundEffectsDefinition(qty_heights=len(heights), heights=heights)
        assert model.heights == heights

    def test_qty_heights_mismatch_raises(self):
        with pytest.raises(ValidationError, match="must equal"):
            GroundEffectsDefinition(qty_heights=4, heights=[100.0, 200.0, 300.0, 400.0, 500.0])


# ──────────────────────────────────────────────────────────────────────────────
# SymmetricFlap
# ──────────────────────────────────────────────────────────────────────────────

class TestSymmetricFlap:
    def test_happy_path(self):
        model = SymmetricFlap(
            flap_type=FlapType.PLAIN,
            nose_type=NoseType.ROUND,
            balance_chord_ratio=0.1,
        )
        assert model.flap_type == FlapType.PLAIN
        assert model.nose_type == NoseType.ROUND
        assert model.balance_chord_ratio == pytest.approx(0.1, abs=1e-6)

    @pytest.mark.parametrize("field", [
        "balance_chord_ratio", "hinge_thickness_to_chord_ratio", "jet_efflux",
    ])
    def test_non_negative_fields_reject_negative(self, field):
        with pytest.raises(ValidationError, match="not be negative|must not be"):
            SymmetricFlap(**{field: -0.01})

    @pytest.mark.parametrize("field", [
        "balance_chord_ratio", "hinge_thickness_to_chord_ratio", "jet_efflux",
    ])
    def test_non_negative_fields_accept_zero(self, field):
        model = SymmetricFlap(**{field: 0.0})
        assert getattr(model, field) == pytest.approx(0.0, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# AsymmetricControl
# ──────────────────────────────────────────────────────────────────────────────

class TestAsymmetricControl:
    def test_happy_path(self):
        model = AsymmetricControl(
            control_type=ControlType.AILERON,
            inboard_aileron_chord_ratio=0.1,
            outboard_span_ratio=0.4,
        )
        assert model.control_type == ControlType.AILERON
        assert model.inboard_aileron_chord_ratio == pytest.approx(0.1, abs=1e-6)

    @pytest.mark.parametrize("field", [
        "inboard_aileron_chord_ratio", "outboard_aileron_chord_ratio",
        "inboard_span_ratio", "outboard_span_ratio", "hingeline_chord_ratio",
    ])
    def test_non_negative_fields_reject_negative(self, field):
        with pytest.raises(ValidationError, match="not be negative|must not be"):
            AsymmetricControl(**{field: -0.1})


# ──────────────────────────────────────────────────────────────────────────────
# Body
# ──────────────────────────────────────────────────────────────────────────────

class TestBody:
    def test_happy_path(self):
        model = Body(
            qty_cross_sections=3,
            stations=[0.1, 0.5, 0.9],
            cross_sectional_areas=[1.0, 2.0, 3.0],
            cross_sectional_perimeters=[4.0, 5.0, 6.0],
            max_halfbredth=[7.0, 8.0, 9.0],
            crown_line=[10.0, 11.0, 12.0],
            keel_line=[13.0, 14.0, 15.0],
            nose_type=BodyShape.CONICAL,
        )
        assert model.qty_cross_sections == 3
        assert model.nose_type == BodyShape.CONICAL
        assert len(model.stations) == 3

    def test_negative_stations_rejected(self):
        with pytest.raises(ValidationError, match="non-negative"):
            Body(
                qty_cross_sections=2,
                stations=[-0.1, 0.5],
                cross_sectional_areas=[1.0, 2.0],
                cross_sectional_perimeters=[1.0, 2.0],
                max_halfbredth=[1.0, 2.0],
                crown_line=[1.0, 2.0],
                keel_line=[1.0, 2.0],
            )

    def test_list_length_mismatch_raises(self):
        with pytest.raises(ValidationError, match="must match qty_cross_sections"):
            Body(
                qty_cross_sections=3,
                stations=[0.1, 0.5],
                cross_sectional_areas=[1.0, 2.0],
                cross_sectional_perimeters=[1.0, 2.0],
                max_halfbredth=[1.0, 2.0],
                crown_line=[1.0, 2.0],
                keel_line=[1.0, 2.0],
            )

    def test_negative_cross_sectional_areas_rejected(self):
        with pytest.raises(ValidationError, match="non-negative"):
            Body(
                qty_cross_sections=2,
                stations=[0.1, 0.5],
                cross_sectional_areas=[-1.0, 2.0],
                cross_sectional_perimeters=[1.0, 2.0],
                max_halfbredth=[1.0, 2.0],
                crown_line=[1.0, 2.0],
                keel_line=[1.0, 2.0],
            )


# ──────────────────────────────────────────────────────────────────────────────
# LowAspectRatioWingBody
# ──────────────────────────────────────────────────────────────────────────────

class TestLowAspectRatioWingBody:
    def test_happy_path(self):
        model = LowAspectRatioWingBody(
            body_centroid_height=0.0,
            reference_area=1.2,
            base_aft_of_lifting_surface=True,
        )
        assert model.body_centroid_height == pytest.approx(0.0, abs=1e-6)
        assert model.reference_area == pytest.approx(1.2, abs=1e-6)
        assert model.base_aft_of_lifting_surface is True

    @pytest.mark.parametrize("field", [
        "body_centroid_height", "reference_area", "sharpness", "frontal_area",
        "aspect_ratio", "effective_radius", "wetted_area", "base_area",
    ])
    def test_numeric_fields_reject_negative(self, field):
        with pytest.raises(ValidationError, match="must not be negative|Numeric values"):
            LowAspectRatioWingBody(**{field: -0.1})

    @pytest.mark.parametrize("field", [
        "body_centroid_height", "reference_area", "sharpness", "frontal_area",
    ])
    def test_zero_accepted(self, field):
        model = LowAspectRatioWingBody(**{field: 0.0})
        assert getattr(model, field) == pytest.approx(0.0, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# TransverseJetControl
# ──────────────────────────────────────────────────────────────────────────────

class TestTransverseJetControl:
    def test_happy_path(self):
        model = TransverseJetControl(
            qty_time=3,
            time=[0.1, 0.2, 0.3],
            control_force=[1.0, 1.1, 1.2],
            altitudes=[5.0, 5.1, 5.2],
        )
        assert model.qty_time == 3
        assert len(model.time) == 3
        assert len(model.control_force) == 3

    def test_time_list_length_mismatch_raises(self):
        with pytest.raises(ValidationError, match="must match qty_time"):
            TransverseJetControl(qty_time=3, time=[0.1, 0.2])

    def test_control_force_list_length_mismatch_raises(self):
        with pytest.raises(ValidationError, match="must match qty_time"):
            TransverseJetControl(qty_time=2, time=[0.1, 0.2], control_force=[1.0])

    def test_altitudes_list_length_mismatch_raises(self):
        with pytest.raises(ValidationError, match="must match qty_time"):
            TransverseJetControl(qty_time=2, time=[0.1, 0.2], control_force=[1.0, 1.1], altitudes=[5.0])


# ──────────────────────────────────────────────────────────────────────────────
# HypersonicFlapControl
# ──────────────────────────────────────────────────────────────────────────────

class TestHypersonicFlapControl:
    def test_happy_path(self):
        model = HypersonicFlapControl(
            altitude=10.0, qty_deflections=2, deflections=[0.1, 0.2]
        )
        assert model.altitude == pytest.approx(10.0, abs=1e-6)
        assert model.qty_deflections == 2
        assert len(model.deflections) == 2

    def test_deflections_length_mismatch_raises(self):
        with pytest.raises(ValidationError, match="must match qty_deflections"):
            HypersonicFlapControl(qty_deflections=3, deflections=[0.1, 0.2])


# ──────────────────────────────────────────────────────────────────────────────
# PropellerPowerProperties
# ──────────────────────────────────────────────────────────────────────────────

class TestPropellerPowerProperties:
    def test_happy_path(self):
        model = PropellerPowerProperties(
            thrust_incidence_angle=5.0, qty_engines=2, prop_radius=1.5
        )
        assert model.thrust_incidence_angle == pytest.approx(5.0, abs=1e-6)
        assert model.qty_engines == 2
        assert model.prop_radius == pytest.approx(1.5, abs=1e-6)

    @pytest.mark.parametrize("field", [
        "thrust_incidence_angle", "prop_radius", "thrust_coefficient", "normal_force_factor",
    ])
    def test_non_negative_fields_reject_negative(self, field):
        with pytest.raises(ValidationError, match="non-negative|must be non"):
            PropellerPowerProperties(**{field: -0.01})

    @pytest.mark.parametrize("field", [
        "thrust_incidence_angle", "prop_radius", "thrust_coefficient", "normal_force_factor",
    ])
    def test_non_negative_fields_accept_zero(self, field):
        model = PropellerPowerProperties(**{field: 0.0})
        assert getattr(model, field) == pytest.approx(0.0, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# JetPowerProperties
# ──────────────────────────────────────────────────────────────────────────────

class TestJetPowerProperties:
    def test_happy_path(self):
        model = JetPowerProperties(
            qty_engines=2, thrust_incidence_angle=5.0, inlet_area=1.2
        )
        assert model.qty_engines == 2
        assert model.thrust_incidence_angle == pytest.approx(5.0, abs=1e-6)
        assert model.inlet_area == pytest.approx(1.2, abs=1e-6)

    @pytest.mark.parametrize("field", [
        "thrust_incidence_angle", "thrust_coefficient", "inlet_area",
        "exhaust_diameter", "exhaust_exit_velocity", "exhaust_total_pressure",
    ])
    def test_non_negative_fields_reject_negative(self, field):
        with pytest.raises(ValidationError, match="non-negative|must be non"):
            JetPowerProperties(**{field: -0.01})


# ──────────────────────────────────────────────────────────────────────────────
# AerodynamicsData
# ──────────────────────────────────────────────────────────────────────────────

class TestAerodynamicsData:
    def test_happy_path(self):
        model = AerodynamicsData(
            CLalpha_body=[0.1, 0.2],
            CMalpha_body=[0.01, 0.02],
            CD_body=[0.001, 0.002],
            CL_body=[0.1, 0.2],
            CM_body=[0.01, 0.02],
            CLAC=[0.1, 0.2], CMAC=[0.01, 0.02], CDC=[0.001, 0.002],
            CLC=[0.1, 0.2], CMC=[0.01, 0.02],
            CLAW=[0.1, 0.2], CMAW=[0.01, 0.02], CDW=[0.001, 0.002],
            CLW=[0.1, 0.2], CMW=[0.01, 0.02],
            CLAH=[0.1, 0.2], CMAH=[0.01, 0.02], CDH=[0.001, 0.002],
            CLH=[0.1, 0.2], CMH=[0.01, 0.02],
            CLAV=[0.1, 0.2], CMAV=[0.01, 0.02], CDV=[0.001, 0.002],
            CLV=[0.1, 0.2], CMV=[0.01, 0.02],
            CLAF=[0.1, 0.2], CMAF=[0.01, 0.02], CDF=[0.001, 0.002],
            CLF=[0.1, 0.2], CMF=[0.01, 0.02],
            CLAWB=[0.1, 0.2], CMAWB=[0.01, 0.02], CDWB=[0.001, 0.002],
            CLWB=[0.1, 0.2], CMWB=[0.01, 0.02],
            DEODA=[0.1, 0.2], EPSLON=[0.1, 0.2], QHOQINF=[0.1, 0.2],
        )
        assert model.CLalpha_body == [0.1, 0.2]
        assert len(model.CLAWB) == 2

    def test_negative_coefficient_rejected(self):
        with pytest.raises(ValidationError, match="non-negative"):
            AerodynamicsData(CLalpha_body=[-0.1, 0.2], CMalpha_body=[0.01, 0.02])

    def test_inconsistent_list_lengths_rejected(self):
        with pytest.raises(ValidationError, match="same length"):
            AerodynamicsData(
                CLalpha_body=[0.1, 0.2, 0.3],
                CMalpha_body=[0.01, 0.02],
            )
