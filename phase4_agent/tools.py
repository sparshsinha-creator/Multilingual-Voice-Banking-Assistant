"""
Phase 4 of 7 -- Atal Pension Yojana eligibility rules engine

Eligibility is a fixed, publicly documented rule (age band, account
requirement, income-tax cutoff), not a fact for the LLM to look up or
guess -- so it's implemented as a deterministic tool the agent calls,
instead of being routed through RAG retrieval.
"""


def check_apy_eligibility(
    age: int,
    has_bank_or_post_office_account: bool,
    is_income_tax_payer: bool,
    enrolled_on_or_before_30_sept_2022: bool,
) -> dict:
    """Apply Atal Pension Yojana eligibility rules and return {"eligible", "reason"}."""
    if not has_bank_or_post_office_account:
        return {
            "eligible": False,
            "reason": "A bank or post office savings account is required for Atal Pension Yojana.",
        }

    if enrolled_on_or_before_30_sept_2022:
        return {
            "eligible": True,
            "reason": (
                "Enrolled on or before 30 September 2022, so the existing Atal Pension "
                "Yojana enrollment continues regardless of income-tax-payer status."
            ),
        }

    if not (18 <= age <= 40):
        return {
            "eligible": False,
            "reason": "Age must be between 18 and 40 (inclusive) to newly join Atal Pension Yojana.",
        }

    if is_income_tax_payer:
        return {
            "eligible": False,
            "reason": (
                "Income-tax payers are not permitted to newly join Atal Pension Yojana "
                "on or after 1 October 2022."
            ),
        }

    return {
        "eligible": True,
        "reason": (
            "Age is within 18-40, a bank or post office account is held, and the applicant "
            "is not an income-tax payer, so new enrollment in Atal Pension Yojana is allowed."
        ),
    }
