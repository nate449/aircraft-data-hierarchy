import unittest
from pydantic import ValidationError
from typing import List
from aircraft_data_hierarchy.work_breakdown_structure.airframe.airframe_geometry import (
    CrossSection, Body, Point, Polyline, Mesh, Airfoil, Spline, LiftingSurface, Loft, String, Boolean, Float, Integer, Metadata
)

class TestPydanticModels(unittest.TestCase):

    def test_cross_section(self):
        # Test valid data
        points = [
            Point(x=0.0, y=0.0, z=0.0),
            Point(x=0.1, y=0.1, z=0.1),
            Point(x=0.5, y=0.5, z=0.5),
            Point(x=0.8, y=0.8, z=0.8),
            Point(x=1.0, y=1.0, z=1.0)
        ]
        upper_curve = Spline(points=points)
        lower_curve = Spline(points=points)
        data = {
            "station": 0.5,
            "upper_curve": upper_curve,
            "lower_curve": lower_curve
        }
        model = CrossSection(**data)
        self.assertEqual(model.station, 0.5)
        self.assertEqual(model.upper_curve, upper_curve)
        self.assertEqual(model.lower_curve, lower_curve)

        # Test missing both curves
        data = {
            "station": 0.5
        }
        with self.assertRaises(ValidationError):
            CrossSection(**data)

    def test_body_geometry(self):
        # Test valid data
        points = [
            Point(x=0.0, y=0.0, z=0.0),
            Point(x=0.1, y=0.1, z=0.1),
            Point(x=0.5, y=0.5, z=0.5),
            Point(x=0.8, y=0.8, z=0.8),
            Point(x=1.0, y=1.0, z=1.0)
        ]
        reference_axis = Spline(points=points)
        cross_section = CrossSection(station=0.5, upper_curve=Spline(points=points))
        data = {
            "reference_axis": reference_axis,
            "cross_sections": [cross_section]
        }
        model = Body(**data)
        self.assertEqual(model.reference_axis, reference_axis)
        self.assertEqual(model.cross_sections, [cross_section])

        # Test missing cross_sections
        data["cross_sections"] = []
        with self.assertRaises(ValidationError):
            Body(**data)

    def test_lifting_surface_geometry(self):
        # Test valid data
        points = [
            Point(x=0.0, y=0.0, z=0.0),
            Point(x=0.1, y=0.1, z=0.1),
            Point(x=0.5, y=0.5, z=0.5),
            Point(x=0.8, y=0.8, z=0.8),
            Point(x=1.0, y=1.0, z=1.0)
        ]
        leading_edge_spline = Spline(points=points)
        trailing_edge_spline = Spline(points=points)
        airfoil_section = Airfoil(spline=Spline(points=points))
        data = {
            "leading_edge_spline": leading_edge_spline,
            "trailing_edge_spline": trailing_edge_spline,
            "airfoil_sections": [airfoil_section]
        }
        model = LiftingSurface(**data)
        self.assertEqual(model.leading_edge_spline, leading_edge_spline)
        self.assertEqual(model.trailing_edge_spline, trailing_edge_spline)
        self.assertEqual(model.airfoil_sections, [airfoil_section])

        # Test missing airfoil_sections
        data["airfoil_sections"] = []
        with self.assertRaises(ValidationError):
            LiftingSurface(**data)

class TestPoint(unittest.TestCase):
    def test_point_creation(self):
        point = Point(x=1.0, y=2.0, z=3.0)
        self.assertEqual(point.x, 1.0)
        self.assertEqual(point.y, 2.0)
        self.assertEqual(point.z, 3.0)

    def test_point_distance(self):
        point1 = Point(x=0.0, y=0.0, z=0.0)
        point2 = Point(x=1.0, y=1.0, z=1.0)
        self.assertAlmostEqual(point1.distance_to(point2), 1.732, places=3)

class TestPolyline(unittest.TestCase):
    def test_polyline_creation(self):
        points = [Point(x=0.0, y=0.0, z=0.0), Point(x=1.0, y=1.0, z=1.0)]
        polyline = Polyline(points=points)
        self.assertEqual(len(polyline.points), 2)

    def test_polyline_length(self):
        points = [Point(x=0.0, y=0.0, z=0.0), Point(x=1.0, y=1.0, z=1.0)]
        polyline = Polyline(points=points)
        self.assertAlmostEqual(polyline.length(), 1.732, places=3)

    def test_polyline_simplify(self):
        points = [
            Point(x=0.0, y=0.0, z=0.0),
            Point(x=0.5, y=0.5, z=0.5),
            Point(x=1.0, y=1.0, z=1.0)
        ]
        polyline = Polyline(points=points)
        simplified_polyline = polyline.simplify(tolerance=0.1)
        self.assertEqual(len(simplified_polyline.points), 2)

class TestMesh(unittest.TestCase):
    def test_mesh_creation(self):
        points = [Point(x=0.0, y=0.0, z=0.0), Point(x=1.0, y=1.0, z=1.0)]
        polyline = Polyline(points=points)
        mesh = Mesh(polylines=[polyline])
        self.assertEqual(len(mesh.polylines), 1)

    # TODO: Fix Issue with unexpected is_manifold() results
    # def test_mesh_is_manifold(self):
    #     points = [
    #         Point(x=0.0, y=0.0, z=0.0),
    #         Point(x=1.0, y=0.0, z=0.0),
    #         Point(x=1.0, y=1.0, z=0.0),
    #         Point(x=0.0, y=1.0, z=0.0)
    #     ]
    #     polyline = Polyline(points=points)
    #     mesh = Mesh(polylines=[polyline])
    #     self.assertTrue(mesh.is_manifold())

    def test_mesh_calculate_volume(self):
        points = [
            Point(x=0.0, y=0.0, z=0.0),
            Point(x=1.0, y=0.0, z=0.0),
            Point(x=1.0, y=1.0, z=0.0),
            Point(x=0.0, y=1.0, z=0.0)
        ]
        polyline = Polyline(points=points)
        mesh = Mesh(polylines=[polyline])
        self.assertAlmostEqual(mesh.calculate_volume(), 0.0, places=3)

class TestSpline(unittest.TestCase):
    def test_spline_creation(self):
        points = [Point(x=0.0, y=0.0, z=0.0), Point(x=1.0, y=1.0, z=1.0),
                  Point(x=2.0, y=2.0, z=2.0),  Point(x=3.0, y=3.0, z=3.0)]
        spline = Spline(points=points, degree=3)
        self.assertEqual(len(spline.points), 4)
        self.assertEqual(spline.degree, 3)

    def test_spline_validation(self):
        points = [Point(x=0.0, y=0.0, z=0.0)]
        with self.assertRaises(ValidationError):
            Spline(points=points, degree=3)

class TestLoft(unittest.TestCase):
    def test_loft_creation(self):
        points1 = [Point(x=0.0, y=0.0, z=0.0), Point(x=1.0, y=1.0, z=1.0),
                   Point(x=2.0, y=2.0, z=2.0), Point(x=3.0, y=3.0, z=3.0)]
        points2 = [Point(x=0.0, y=0.0, z=1.0), Point(x=1.0, y=1.0, z=2.0),
                   Point(x=2.0, y=2.0, z=3.0), Point(x=3.0, y=3.0, z=4.0)]
        spline1 = Spline(points=points1, degree=4)
        spline2 = Spline(points=points2, degree=4)
        loft = Loft(splines=[spline1, spline2], num_samples=10)
        self.assertEqual(len(loft.splines), 2)
        self.assertEqual(loft.num_samples, 10)

    def test_loft_calculate_surface(self):
        points1 = [Point(x=0.0, y=0.0, z=0.0), Point(x=1.0, y=1.0, z=1.0),
                   Point(x=2.0, y=2.0, z=2.0), Point(x=3.0, y=3.0, z=3.0)]
        points2 = [Point(x=0.0, y=0.0, z=1.0), Point(x=1.0, y=1.0, z=2.0),
                   Point(x=2.0, y=2.0, z=3.0), Point(x=3.0, y=3.0, z=4.0)]
        spline1 = Spline(points=points1, degree=4)
        spline2 = Spline(points=points2, degree=4)
        loft = Loft(splines=[spline1, spline2], num_samples=10)
        surface = loft.calculate_surface()
        self.assertEqual(len(surface), 40)  # 2 splines * 10 samples

class TestString(unittest.TestCase):
    def test_string_creation(self):
        metadata = Metadata(key="example_key", value="example_value")
        string = String(value="test", default="default", metadata=metadata)
        self.assertEqual(string.value, "test")
        self.assertEqual(string.default, "default")

    def test_string_validation(self):
        with self.assertRaises(ValidationError):
            String(value="")

class TestBoolean(unittest.TestCase):
    def test_boolean_creation(self):
        metadata = Metadata(key="example_key", value="example_value")
        boolean = Boolean(value=True, default=False, metadata=metadata)
        self.assertTrue(boolean.value)
        self.assertFalse(boolean.default)

    def test_boolean_validation(self):
        with self.assertRaises(ValidationError):
            Boolean(value="not a boolean")

class TestFloat(unittest.TestCase):
    def test_float_creation(self):
        metadata = Metadata(key="example_key", value="example_value")
        float_var = Float(value=1.23, default=0.0, metadata=metadata)
        self.assertAlmostEqual(float_var.value, 1.23)
        self.assertAlmostEqual(float_var.default, 0.0)

    def test_float_validation(self):
        with self.assertRaises(ValidationError):
            Float(value="not a float")

class TestInteger(unittest.TestCase):
    def test_integer_creation(self):
        metadata = Metadata(key="example_key", value="example_value")
        integer = Integer(value=123, default=0, metadata=metadata)
        self.assertEqual(integer.value, 123)
        self.assertEqual(integer.default, 0)

    def test_integer_validation(self):
        with self.assertRaises(ValidationError):
            Integer(value="not an integer")

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# ---------------------------------------------------------------------------
# Pytest-style boundary, error-path, and cross-field tests
# ---------------------------------------------------------------------------
import pytest
import math

from aircraft_data_hierarchy.work_breakdown_structure.airframe.airframe_geometry import (
    Boolean, Float, Integer, String, Point, Polyline, Spline, Mesh,
    Loft, Airfoil, ReferenceAxis, CrossSection, Body, LiftingSurface,
)


def _pts(n: int) -> List[Point]:
    """Return *n* distinct points along the line x=y=z."""
    return [Point(x=float(i), y=float(i), z=float(i)) for i in range(n)]


def _spline(n: int = 4, degree: int = 3) -> Spline:
    return Spline(points=_pts(n), degree=degree)


def _polyline(n: int = 3) -> Polyline:
    return Polyline(points=_pts(n))


# ── Boolean.validate_default  (lines 94-100) ─────────────────────────────

class TestBooleanDefaultConversion:

    @pytest.mark.parametrize("raw,expected", [
        ("true", True), ("True", True), ("TRUE", True),
        ("1", True), ("t", True), ("y", True), ("yes", True),
        ("false", False), ("False", False), ("FALSE", False),
        ("0", False), ("f", False), ("n", False), ("no", False),
    ])
    def test_string_to_bool_accepted(self, raw, expected):
        b = Boolean(value=True, default=raw)
        assert b.default is expected

    @pytest.mark.parametrize("bad", ["maybe", "2", "nope", "yep", "oui"])
    def test_string_to_bool_rejected(self, bad):
        with pytest.raises(ValidationError, match="Invalid string value for a boolean conversion"):
            Boolean(value=True, default=bad)


# ── Float.validate_default  (lines 145-148) ──────────────────────────────

class TestFloatDefaultConversion:

    @pytest.mark.parametrize("raw,expected", [
        ("3.14", 3.14), ("-1.5", -1.5), ("0", 0.0), ("1e2", 100.0),
    ])
    def test_string_to_float_accepted(self, raw, expected):
        f = Float(value=0.0, default=raw)
        assert f.default == pytest.approx(expected)

    @pytest.mark.parametrize("bad", ["abc", "one-point-five"])
    def test_string_to_float_rejected(self, bad):
        with pytest.raises(ValidationError, match="Invalid string value for a float conversion"):
            Float(value=0.0, default=bad)


# ── Integer.validate_default  (lines 193-196) ────────────────────────────

class TestIntegerDefaultConversion:

    @pytest.mark.parametrize("raw,expected", [
        ("42", 42), ("-7", -7), ("0", 0),
    ])
    def test_string_to_int_accepted(self, raw, expected):
        i = Integer(value=0, default=raw)
        assert i.default == expected

    @pytest.mark.parametrize("bad", ["abc", "3.14", "one"])
    def test_string_to_int_rejected(self, bad):
        with pytest.raises(ValidationError, match="Invalid string value for an integer conversion"):
            Integer(value=0, default=bad)


# ── Point.validate_coordinate  (line 239) ────────────────────────────────

class TestPointCoordinateValidation:

    @pytest.mark.parametrize("kwargs", [
        {"x": float("inf"), "y": 0.0, "z": 0.0},
        {"x": 0.0, "y": float("-inf"), "z": 0.0},
        {"x": 0.0, "y": 0.0, "z": float("nan")},
    ])
    def test_non_finite_coordinate_rejected(self, kwargs):
        with pytest.raises(ValidationError, match="Coordinate values must be finite"):
            Point(**kwargs)

    def test_distance_to_known_geometry(self):
        """3-4-5 right triangle in 3D."""
        a = Point(x=0.0, y=0.0, z=0.0)
        b = Point(x=3.0, y=4.0, z=0.0)
        assert a.distance_to(b) == pytest.approx(5.0)


# ── Point.__hash__  (line 257) ───────────────────────────────────────────

class TestPointHash:

    def test_equal_points_hash_equally(self):
        p1 = Point(x=1.0, y=2.0, z=3.0)
        p2 = Point(x=1.0, y=2.0, z=3.0)
        assert hash(p1) == hash(p2)

    def test_set_deduplication(self):
        p1 = Point(x=1.0, y=2.0, z=3.0)
        p2 = Point(x=1.0, y=2.0, z=3.0)
        p3 = Point(x=9.0, y=8.0, z=7.0)
        assert len({p1, p2, p3}) == 2


# ── Polyline validators / methods  (lines 292, 301, 328, 345) ────────────

class TestPolylineBoundary:

    @pytest.mark.parametrize("n", [0, 1])
    def test_fewer_than_two_points_rejected(self, n):
        with pytest.raises(ValidationError, match="at least two points"):
            Polyline(points=_pts(n))

    def test_add_point_extends_path(self):
        poly = _polyline(2)
        poly.add_point(Point(x=10.0, y=10.0, z=10.0))
        assert len(poly.points) == 3

    def test_simplify_two_points_unchanged(self):
        poly = Polyline(points=_pts(2))
        result = poly.simplify(tolerance=0.01)
        assert len(result.points) == 2

    def test_simplify_retains_deviating_point(self):
        pts = [
            Point(x=0.0, y=0.0, z=0.0),
            Point(x=0.5, y=5.0, z=0.0),
            Point(x=1.0, y=0.0, z=0.0),
        ]
        result = Polyline(points=pts).simplify(tolerance=0.01)
        assert len(result.points) == 3

    def test_simplify_removes_collinear_point(self):
        pts = [
            Point(x=0.0, y=0.0, z=0.0),
            Point(x=0.5, y=0.0, z=0.0),
            Point(x=1.0, y=0.0, z=0.0),
        ]
        result = Polyline(points=pts).simplify(tolerance=0.01)
        assert len(result.points) == 2

    def test_length_multi_segment(self):
        pts = [
            Point(x=0.0, y=0.0, z=0.0),
            Point(x=3.0, y=4.0, z=0.0),
            Point(x=3.0, y=4.0, z=12.0),
        ]
        assert Polyline(points=pts).length() == pytest.approx(17.0)


# ── Spline.validate_degree  (line 411) ───────────────────────────────────

class TestSplineBoundary:

    @pytest.mark.parametrize("deg", [0, -1, -100])
    def test_non_positive_degree_rejected(self, deg):
        with pytest.raises(ValidationError, match="positive integer|greater than 0"):
            Spline(points=_pts(10), degree=deg)

    def test_insufficient_points_for_degree(self):
        with pytest.raises(ValidationError, match="At least 4 points are required"):
            Spline(points=_pts(3), degree=3)

    def test_exact_minimum_points_accepted(self):
        s = Spline(points=_pts(4), degree=3)
        assert len(s.points) == 4


# ── Mesh validators / methods  (lines 448, 457, 468-470) ─────────────────

class TestMeshBoundary:

    def test_empty_polylines_rejected(self):
        with pytest.raises(ValidationError, match="at least one polyline"):
            Mesh(polylines=[])

    def test_add_polyline(self):
        mesh = Mesh(polylines=[_polyline()])
        mesh.add_polyline(_polyline(4))
        assert len(mesh.polylines) == 2

    @pytest.mark.parametrize("idx", [-1, 1, 99])
    def test_remove_polyline_bad_index(self, idx):
        mesh = Mesh(polylines=[_polyline()])
        with pytest.raises(IndexError, match="Invalid index"):
            mesh.remove_polyline(idx)

    def test_remove_polyline_valid(self):
        mesh = Mesh(polylines=[_polyline(), _polyline(4)])
        mesh.remove_polyline(0)
        assert len(mesh.polylines) == 1

    def test_calculate_volume_unit_tetrahedron(self):
        """Four triangular faces forming a tetrahedron with known volume = 1/6."""
        a, b, c, d = (
            Point(x=0, y=0, z=0),
            Point(x=1, y=0, z=0),
            Point(x=0, y=1, z=0),
            Point(x=0, y=0, z=1),
        )
        face1 = Polyline(points=[a, b, c])
        face2 = Polyline(points=[a, b, d])
        face3 = Polyline(points=[a, c, d])
        face4 = Polyline(points=[b, c, d])
        mesh = Mesh(polylines=[face1, face2, face3, face4])
        assert mesh.calculate_volume() == pytest.approx(1.0 / 6.0, abs=1e-9)


# ── Loft validators / methods  (lines 593, 597, 615, 624) ────────────────

class TestLoftBoundary:

    def test_single_spline_rejected(self):
        with pytest.raises(ValidationError, match="at least two splines"):
            Loft(splines=[_spline()])

    def test_mismatched_degrees_rejected(self):
        s1 = Spline(points=_pts(4), degree=3)
        s2 = Spline(points=_pts(4), degree=1)
        with pytest.raises(ValidationError, match="same degree"):
            Loft(splines=[s1, s2])

    @pytest.mark.parametrize("n", [0, -1, -100])
    def test_non_positive_num_samples_rejected(self, n):
        with pytest.raises(ValidationError, match="positive integer"):
            Loft(splines=[_spline(), _spline()], num_samples=n)

    def test_add_spline(self):
        loft = Loft(splines=[_spline(), _spline()], num_samples=5)
        loft.add_spline(_spline())
        assert len(loft.splines) == 3

    def test_calculate_surface_interpolation(self):
        """Two axis-aligned splines separated in z; verify midpoint interpolation."""
        pa = [Point(x=float(i), y=0, z=0) for i in range(4)]
        pb = [Point(x=float(i), y=0, z=1) for i in range(4)]
        loft = Loft(
            splines=[Spline(points=pa), Spline(points=pb)],
            num_samples=3,
        )
        surface = loft.calculate_surface()
        assert len(surface) == 12  # 3 samples × 4 pts
        mid_row = surface[4:8]  # t=0.5
        for pt in mid_row:
            assert pt[2] == pytest.approx(0.5)


# ── Airfoil.validate_spline  (line 713) ──────────────────────────────────

class TestAirfoilBoundary:

    def test_none_spline_rejected(self):
        with pytest.raises(ValidationError, match="spline defining the airfoil contour must be provided"):
            Airfoil(spline=None)


# ── LiftingSurface.validate_airfoil_sections ─────────────────────────────

class TestLiftingSurfaceBoundary:

    def test_empty_airfoil_sections_rejected(self):
        with pytest.raises(ValidationError, match="At least one Airfoil must be provided"):
            LiftingSurface(airfoil_sections=[])


# ── CrossSection.validate_curves  ────────────────────────────────────────

class TestCrossSectionBoundary:

    def test_no_curves_rejected(self):
        with pytest.raises(ValidationError, match="(?i)at least one of the upper or lower curve"):
            CrossSection(station=0.5)

    def test_station_out_of_range(self):
        with pytest.raises(ValidationError):
            CrossSection(station=1.5, upper_curve=_spline())

    def test_only_upper_accepted(self):
        cs = CrossSection(station=0.0, upper_curve=_spline())
        assert cs.lower_curve is None

    def test_only_lower_accepted(self):
        cs = CrossSection(station=1.0, lower_curve=_spline())
        assert cs.upper_curve is None


# ── Body.validate_cross_sections  ────────────────────────────────────────

class TestBodyBoundary:

    def test_empty_cross_sections_rejected(self):
        with pytest.raises(ValidationError, match="At least one CrossSection must be provided"):
            Body(cross_sections=[])


# ── String.validate_value_not_empty  ─────────────────────────────────────

class TestStringBoundary:

    @pytest.mark.parametrize("bad", ["   ", "\t", "\n"])
    def test_whitespace_only_rejected(self, bad):
        with pytest.raises(ValidationError, match="cannot be empty"):
            String(value=bad)


# ── ReferenceAxis — broken _registry (Pydantic v2 bug) ──────────────────
# _registry is declared as a plain class attribute with underscore prefix,
# but Pydantic v2 treats it as a ModelPrivateAttr.  This means:
#   • validate_and_register (mode="before") hits `name in cls._registry`
#     where cls._registry is a ModelPrivateAttr, raising TypeError.
#   • All downstream validators and __init__ registration are unreachable.
# The tests below document this behaviour; the ~15 uncovered lines inside
# ReferenceAxis are blocked by this bug and cannot be reached without a
# source-code fix.

class TestReferenceAxisBrokenRegistry:

    def test_construction_fails_due_to_private_attr_registry(self):
        """Creating ANY ReferenceAxis raises TypeError because
        `name in cls._registry` fails on a ModelPrivateAttr descriptor."""
        with pytest.raises(TypeError, match="not iterable"):
            ReferenceAxis(name="axis", points=_pts(2))
