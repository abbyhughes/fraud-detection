import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from risk_rules import label_risk, score_transaction
from features import build_model_frame
from analyze_fraud import load_inputs, score_transactions, summarize_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_tx(**overrides):
    """Minimal clean transaction that scores 0 before overrides."""
    tx = {
        "device_risk_score": 5,
        "is_international": 0,
        "amount_usd": 100.0,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# ---------------------------------------------------------------------------
# label_risk
# ---------------------------------------------------------------------------

class TestLabelRisk:
    def test_low_boundary(self):
        assert label_risk(0) == "low"
        assert label_risk(29) == "low"

    def test_medium_boundary(self):
        assert label_risk(30) == "medium"
        assert label_risk(59) == "medium"

    def test_high_boundary(self):
        assert label_risk(60) == "high"
        assert label_risk(100) == "high"


# ---------------------------------------------------------------------------
# score_transaction — individual signals
# ---------------------------------------------------------------------------

class TestDeviceRiskScore:
    def test_low_device_risk_no_points(self):
        assert score_transaction(base_tx(device_risk_score=5)) == 0

    def test_moderate_device_risk_adds_points(self):
        low = score_transaction(base_tx(device_risk_score=5))
        mid = score_transaction(base_tx(device_risk_score=50))
        assert mid > low

    def test_high_device_risk_adds_more_than_moderate(self):
        mid = score_transaction(base_tx(device_risk_score=50))
        high = score_transaction(base_tx(device_risk_score=80))
        assert high > mid

    def test_high_device_risk_exact_delta(self):
        baseline = score_transaction(base_tx(device_risk_score=5))
        scored = score_transaction(base_tx(device_risk_score=75))
        assert scored - baseline == 25

    def test_moderate_device_risk_exact_delta(self):
        baseline = score_transaction(base_tx(device_risk_score=5))
        scored = score_transaction(base_tx(device_risk_score=55))
        assert scored - baseline == 10


class TestInternational:
    def test_international_raises_score(self):
        domestic = score_transaction(base_tx(is_international=0))
        intl = score_transaction(base_tx(is_international=1))
        assert intl > domestic

    def test_international_exact_delta(self):
        domestic = score_transaction(base_tx(is_international=0))
        intl = score_transaction(base_tx(is_international=1))
        assert intl - domestic == 15


class TestAmount:
    def test_small_amount_no_points(self):
        assert score_transaction(base_tx(amount_usd=100)) == 0

    def test_medium_amount_adds_points(self):
        low = score_transaction(base_tx(amount_usd=100))
        mid = score_transaction(base_tx(amount_usd=600))
        assert mid - low == 10

    def test_large_amount_adds_most_points(self):
        low = score_transaction(base_tx(amount_usd=100))
        high = score_transaction(base_tx(amount_usd=1500))
        assert high - low == 25

    def test_amount_threshold_boundary(self):
        just_under = score_transaction(base_tx(amount_usd=999.99))
        at_threshold = score_transaction(base_tx(amount_usd=1000))
        assert at_threshold > just_under


class TestVelocity:
    def test_low_velocity_no_points(self):
        assert score_transaction(base_tx(velocity_24h=1)) == 0

    def test_medium_velocity_adds_points(self):
        low = score_transaction(base_tx(velocity_24h=1))
        mid = score_transaction(base_tx(velocity_24h=4))
        assert mid > low

    def test_high_velocity_adds_more_than_medium(self):
        mid = score_transaction(base_tx(velocity_24h=4))
        high = score_transaction(base_tx(velocity_24h=8))
        assert high > mid

    def test_high_velocity_exact_delta(self):
        baseline = score_transaction(base_tx(velocity_24h=1))
        scored = score_transaction(base_tx(velocity_24h=6))
        assert scored - baseline == 20

    def test_medium_velocity_exact_delta(self):
        baseline = score_transaction(base_tx(velocity_24h=1))
        scored = score_transaction(base_tx(velocity_24h=3))
        assert scored - baseline == 5


class TestFailedLogins:
    def test_no_failed_logins_no_points(self):
        assert score_transaction(base_tx(failed_logins_24h=0)) == 0

    def test_few_failed_logins_adds_points(self):
        none = score_transaction(base_tx(failed_logins_24h=0))
        some = score_transaction(base_tx(failed_logins_24h=3))
        assert some - none == 10

    def test_many_failed_logins_adds_most_points(self):
        none = score_transaction(base_tx(failed_logins_24h=0))
        many = score_transaction(base_tx(failed_logins_24h=5))
        assert many - none == 20


class TestPriorChargebacks:
    def test_no_chargebacks_no_points(self):
        assert score_transaction(base_tx(prior_chargebacks=0)) == 0

    def test_one_chargeback_raises_score(self):
        clean = score_transaction(base_tx(prior_chargebacks=0))
        one = score_transaction(base_tx(prior_chargebacks=1))
        assert one > clean

    def test_two_or_more_chargebacks_raises_score_more(self):
        one = score_transaction(base_tx(prior_chargebacks=1))
        two = score_transaction(base_tx(prior_chargebacks=2))
        assert two > one

    def test_one_chargeback_exact_delta(self):
        baseline = score_transaction(base_tx(prior_chargebacks=0))
        scored = score_transaction(base_tx(prior_chargebacks=1))
        assert scored - baseline == 5

    def test_two_chargebacks_exact_delta(self):
        baseline = score_transaction(base_tx(prior_chargebacks=0))
        scored = score_transaction(base_tx(prior_chargebacks=2))
        assert scored - baseline == 20


# ---------------------------------------------------------------------------
# score_transaction — combined signals and clamping
# ---------------------------------------------------------------------------

class TestScoreCombinedAndClamping:
    def test_clean_transaction_scores_zero(self):
        assert score_transaction(base_tx()) == 0

    def test_score_never_exceeds_100(self):
        worst_case = base_tx(
            device_risk_score=85,
            is_international=1,
            amount_usd=2000,
            velocity_24h=10,
            failed_logins_24h=6,
            prior_chargebacks=3,
        )
        assert score_transaction(worst_case) == 100

    def test_score_never_below_zero(self):
        assert score_transaction(base_tx()) >= 0

    def test_high_risk_transaction_labelled_high(self):
        tx = base_tx(
            device_risk_score=80,
            is_international=1,
            velocity_24h=7,
            failed_logins_24h=5,
        )
        assert label_risk(score_transaction(tx)) == "high"

    def test_known_fraud_profile_scores_high(self):
        # Mirrors txn 50011: device=85, intl, velocity=8, logins=7, amount=1400
        tx = base_tx(
            device_risk_score=85,
            is_international=1,
            amount_usd=1400,
            velocity_24h=8,
            failed_logins_24h=7,
            prior_chargebacks=1,
        )
        assert label_risk(score_transaction(tx)) == "high"

    def test_domestic_low_risk_profile_scores_low(self):
        # Mirrors txn 50001: domestic, small amount, no signals
        tx = base_tx(
            device_risk_score=8,
            is_international=0,
            amount_usd=45.20,
            velocity_24h=1,
            failed_logins_24h=0,
            prior_chargebacks=0,
        )
        assert label_risk(score_transaction(tx)) == "low"


# ---------------------------------------------------------------------------
# build_model_frame (features.py)
# ---------------------------------------------------------------------------

class TestBuildModelFrame:
    @pytest.fixture
    def sample_data(self):
        transactions = pd.DataFrame([{
            "transaction_id": 1,
            "account_id": 101,
            "amount_usd": 1200.0,
            "failed_logins_24h": 3,
        }])
        accounts = pd.DataFrame([{
            "account_id": 101,
            "prior_chargebacks": 1,
        }])
        return transactions, accounts

    def test_merge_joins_account_data(self, sample_data):
        transactions, accounts = sample_data
        df = build_model_frame(transactions, accounts)
        assert "prior_chargebacks" in df.columns

    def test_is_large_amount_flag_set(self, sample_data):
        transactions, accounts = sample_data
        df = build_model_frame(transactions, accounts)
        assert df.loc[0, "is_large_amount"] == 1

    def test_is_large_amount_flag_not_set(self, sample_data):
        transactions, accounts = sample_data
        transactions.loc[0, "amount_usd"] = 200.0
        df = build_model_frame(transactions, accounts)
        assert df.loc[0, "is_large_amount"] == 0

    def test_login_pressure_high(self, sample_data):
        transactions, accounts = sample_data
        df = build_model_frame(transactions, accounts)
        assert str(df.loc[0, "login_pressure"]) == "high"

    def test_login_pressure_none(self, sample_data):
        transactions, accounts = sample_data
        transactions.loc[0, "failed_logins_24h"] = 0
        df = build_model_frame(transactions, accounts)
        assert str(df.loc[0, "login_pressure"]) == "none"

    def test_missing_account_keeps_transaction(self):
        transactions = pd.DataFrame([{
            "transaction_id": 99,
            "account_id": 999,
            "amount_usd": 50.0,
            "failed_logins_24h": 0,
        }])
        accounts = pd.DataFrame(columns=["account_id", "prior_chargebacks"])
        df = build_model_frame(transactions, accounts)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# End-to-end pipeline metrics
# ---------------------------------------------------------------------------

class TestPipelineMetrics:
    @pytest.fixture(scope="class")
    def pipeline_output(self):
        accounts, transactions, chargebacks = load_inputs()
        scored = score_transactions(transactions, accounts)
        summary = summarize_results(scored, chargebacks)
        return scored, summary, chargebacks

    def test_all_transactions_are_scored(self, pipeline_output):
        scored, _, _ = pipeline_output
        assert scored["risk_score"].notna().all()

    def test_all_scores_in_valid_range(self, pipeline_output):
        scored, _, _ = pipeline_output
        assert scored["risk_score"].between(0, 100).all()

    def test_all_labels_are_valid(self, pipeline_output):
        scored, _, _ = pipeline_output
        assert set(scored["risk_label"]).issubset({"low", "medium", "high"})

    def test_confirmed_chargebacks_not_in_low_risk(self, pipeline_output):
        scored, _, chargebacks = pipeline_output
        fraud_ids = set(chargebacks["transaction_id"])
        fraud_scored = scored[scored["transaction_id"].isin(fraud_ids)]
        assert (fraud_scored["risk_label"] == "low").sum() == 0, (
            "Confirmed chargeback transactions should not be labelled low risk"
        )

    def test_high_risk_bucket_has_highest_chargeback_rate(self, pipeline_output):
        _, summary, _ = pipeline_output
        rates = summary.set_index("risk_label")["chargeback_rate"]
        assert rates.get("high", 0) > rates.get("medium", 0)
        assert rates.get("medium", 0) >= rates.get("low", 0)

    def test_high_risk_chargeback_rate_is_100_percent(self, pipeline_output):
        _, summary, _ = pipeline_output
        rates = summary.set_index("risk_label")["chargeback_rate"]
        assert rates["high"] == 1.0, (
            "Every transaction in the high-risk bucket should be a known chargeback"
        )

    def test_low_risk_chargeback_rate_is_zero(self, pipeline_output):
        _, summary, _ = pipeline_output
        rates = summary.set_index("risk_label")["chargeback_rate"]
        assert rates.get("low", 0) == 0.0, (
            "No confirmed chargebacks should appear in the low-risk bucket"
        )

    def test_summary_covers_all_transactions(self, pipeline_output):
        scored, summary, _ = pipeline_output
        assert summary["transactions"].sum() == len(scored)

    def test_summary_amounts_match_total(self, pipeline_output):
        scored, summary, _ = pipeline_output
        assert pytest.approx(summary["total_amount_usd"].sum(), rel=1e-6) == scored["amount_usd"].sum()
