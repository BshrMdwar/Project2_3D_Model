import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import explore_dataset as ed

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mock_dataset"


def test_report_generates_without_crash():
    report = ed.generate_report(str(FIXTURE_ROOT))
    assert report["total_structurally_valid"] == 6
    assert report["label_validation"]["failed"] == 1


def test_absent_super_categories_detected():
    report = ed.generate_report(str(FIXTURE_ROOT))
    assert "Table" in report["absent_super_categories"]
    assert "Window" in report["absent_super_categories"]
    assert "Bed" in report["active_super_categories"]


def test_low_confidence_combos_flagged_with_low_threshold():
    report = ed.generate_report(str(FIXTURE_ROOT), min_confidence_samples=2)
    assert len(report["low_confidence_combos"]) > 0


def test_print_report_does_not_crash():
    report = ed.generate_report(str(FIXTURE_ROOT))
    ed.print_report(report)  # just verify no exception


def _run_all():
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
