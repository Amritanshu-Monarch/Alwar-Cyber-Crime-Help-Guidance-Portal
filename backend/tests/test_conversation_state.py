"""Unit tests for Phase 3: User-Oriented Conversation, Session State & Triage.

Verifies:
1. Per-session state tracking (case_type, money_lost, amount, payment_method, platform, steps_already_done).
2. Triage logic: asks ONE short question first when essential info is missing.
3. Deduplication of steps already completed by citizen.
4. Token budget & sentence-boundary trimming ensuring replies never get cut mid-sentence.
"""

from app.agent.runner import trim_to_sentence_boundary
from app.agent.state import (
    SessionCaseState,
    extract_and_update_state,
    get_triage_reply,
    needs_triage_question,
)


def test_state_extraction_financial_fraud():
    state = SessionCaseState(session_id="test-session-1")
    text = "Mere UPI se 25000 rupaye cut gaye hain abhi 10 minute pehle"
    state = extract_and_update_state(text, state, "hinglish")

    assert state.case_type == "financial_fraud"
    assert state.money_lost is True
    assert state.amount == "Rs 25000"
    assert state.payment_method == "UPI"
    assert state.time_since_incident == "within_golden_hour"


def test_state_extraction_completed_steps():
    state = SessionCaseState(session_id="test-session-2")
    text = "Main 1930 par call kar diya tha aur ATM card block karwa diya hai, ab aage kya karu?"
    state = extract_and_update_state(text, state, "hinglish")

    assert "called_1930" in state.steps_already_done
    assert "blocked_card" in state.steps_already_done

    summary = state.to_context_summary()
    assert "called_1930" in summary
    assert "blocked_card" in summary
    assert "Do NOT instruct them to repeat these steps" in summary


def test_triage_question_logic():
    state = SessionCaseState(session_id="test-session-3")
    # Generic money lost without method or timing
    text = "Mere paise cut gaye hain help karo"
    state = extract_and_update_state(text, state, "hinglish")

    assert state.money_lost is True
    assert needs_triage_question(state) is True

    triage_msg = get_triage_reply(state, lang="hinglish")
    assert "1930" in triage_msg
    assert "Paise kitne der pehle kate" in triage_msg
    assert needs_triage_question(state) is False  # Once asked, do not re-ask


def test_sentence_boundary_trimming():
    # Complete sentence remains unchanged
    complete = "Sabse pehle 1930 par call karein. Bank me inform karein."
    assert trim_to_sentence_boundary(complete) == complete

    # Truncated sentence is trimmed back cleanly
    truncated = "Sabse pehle 1930 par call karein. Bank me inform karein aur transaction ID"
    expected = "Sabse pehle 1930 par call karein."
    assert trim_to_sentence_boundary(truncated) == expected

    # Devanagari purna viram (।) support
    hindi_truncated = "तुरंत 1930 पर कॉल करें। अपने बैंक को सूचित करें और साइबर सेल"
    hindi_expected = "तुरंत 1930 पर कॉल करें।"
    assert trim_to_sentence_boundary(hindi_truncated) == hindi_expected
