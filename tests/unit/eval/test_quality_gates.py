"""Tests for Phase 1 quality gates: pass/fail logic and step-aware skipping."""

from __future__ import annotations

import pytest

from bgkit.eval.quality_gates import check_phase1_gates


class TestCheckPhase1Gates:
    def test_all_pass(self):
        gates = check_phase1_gates(
            step="all",
            reconstruction_loss=1.5,
            parse_success_rate=0.9,
            text_repro_loss=2.5,
            description_quality=0.6,
        )
        assert all(g.passed for g in gates)
        assert all(not g.skipped for g in gates)

    def test_all_fail(self):
        gates = check_phase1_gates(
            step="all",
            reconstruction_loss=3.0,
            parse_success_rate=0.5,
            text_repro_loss=4.0,
            description_quality=0.3,
        )
        assert all(not g.passed for g in gates if not g.skipped)

    def test_mixed_pass_fail(self):
        gates = check_phase1_gates(
            step="all",
            reconstruction_loss=1.5,  # pass
            parse_success_rate=0.5,  # fail
            text_repro_loss=2.5,  # pass
            description_quality=0.3,  # fail
        )
        by_name = {g.gate_name: g for g in gates}
        assert by_name["reconstruction_loss"].passed
        assert not by_name["parse_success_rate"].passed
        assert by_name["text_reproduction"].passed
        assert not by_name["description_quality"].passed

    def test_custom_thresholds(self):
        gates = check_phase1_gates(
            step="all",
            reconstruction_loss=1.5,
            parse_success_rate=0.9,
            text_repro_loss=2.5,
            description_quality=0.6,
            max_reconstruction_loss=1.0,  # tighter
            min_parse_success_rate=0.95,  # tighter
        )
        by_name = {g.gate_name: g for g in gates}
        assert not by_name["reconstruction_loss"].passed  # 1.5 > 1.0
        assert not by_name["parse_success_rate"].passed  # 0.9 < 0.95


class TestStepAwareSkipping:
    def test_step1_skips_text_repro_and_description(self):
        gates = check_phase1_gates(
            step="1",
            reconstruction_loss=1.5,
            parse_success_rate=0.9,
        )
        by_name = {g.gate_name: g for g in gates}
        assert not by_name["reconstruction_loss"].skipped
        assert not by_name["parse_success_rate"].skipped
        assert by_name["text_reproduction"].skipped
        assert by_name["description_quality"].skipped

    def test_step2_same_as_step1(self):
        gates = check_phase1_gates(
            step="2",
            reconstruction_loss=1.5,
            parse_success_rate=0.9,
        )
        by_name = {g.gate_name: g for g in gates}
        assert not by_name["reconstruction_loss"].skipped
        assert not by_name["parse_success_rate"].skipped
        assert by_name["text_reproduction"].skipped
        assert by_name["description_quality"].skipped

    def test_step3a_includes_text_repro(self):
        gates = check_phase1_gates(
            step="3a",
            reconstruction_loss=1.5,
            parse_success_rate=0.9,
            text_repro_loss=2.5,
        )
        by_name = {g.gate_name: g for g in gates}
        assert not by_name["text_reproduction"].skipped
        assert by_name["description_quality"].skipped

    def test_step3a_fails_without_text_repro_loss(self):
        """step='3a' with missing text_repro_loss should fail, not skip."""
        gates = check_phase1_gates(
            step="3a",
            reconstruction_loss=1.5,
            parse_success_rate=0.9,
            # text_repro_loss not provided
        )
        by_name = {g.gate_name: g for g in gates}
        assert not by_name["text_reproduction"].skipped
        assert not by_name["text_reproduction"].passed

    def test_step3b_includes_all(self):
        gates = check_phase1_gates(
            step="3b",
            reconstruction_loss=1.5,
            parse_success_rate=0.9,
            text_repro_loss=2.5,
            description_quality=0.6,
        )
        assert all(not g.skipped for g in gates)

    def test_skipped_gates_count_as_passed(self):
        """Skipped gates should not fail the check."""
        gates = check_phase1_gates(
            step="1",
            reconstruction_loss=1.5,
            parse_success_rate=0.9,
        )
        assert all(g.passed for g in gates)

    def test_missing_metric_at_step_all_fails(self):
        """step='all' with missing active metrics should fail, not skip."""
        gates = check_phase1_gates(
            step="all",
            reconstruction_loss=1.5,
            parse_success_rate=0.9,
            # text_repro_loss and description_quality not provided
        )
        by_name = {g.gate_name: g for g in gates}
        assert not by_name["text_reproduction"].skipped
        assert not by_name["text_reproduction"].passed
        assert not by_name["description_quality"].skipped
        assert not by_name["description_quality"].passed

    def test_missing_metric_at_early_step_is_skipped(self):
        """Early steps with missing later-stage metrics should skip, not fail."""
        gates = check_phase1_gates(
            step="1",
            reconstruction_loss=1.5,
            parse_success_rate=0.9,
            # text_repro_loss and description_quality not provided — not active at step 1
        )
        by_name = {g.gate_name: g for g in gates}
        assert by_name["text_reproduction"].skipped
        assert by_name["description_quality"].skipped

    def test_invalid_step_raises(self):
        with pytest.raises(ValueError, match="Unknown step"):
            check_phase1_gates(step="invalid")
