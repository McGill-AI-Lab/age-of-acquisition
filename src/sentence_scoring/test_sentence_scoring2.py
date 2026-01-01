"""
Comprehensive tests for sentence_scoring.score_sentence.

These tests are designed to be robust across different lexical tables:
- They validate *invariants* (stopword skipping, witness eligibility, -1 behavior),
  rather than hard-coding specific scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from sentence_scoring import score_sentence
from sentence_scoring.stopwords import STOPWORDS
from sentence_scoring.simple_tokenizer import simple_tokenize

from lexical_features import aoa, conc, phon, freq


# ----------------------------
# Tiny assertion helpers
# ----------------------------

class TestFail(Exception):
    pass


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise TestFail(msg)


def assert_eq(a, b, msg: str) -> None:
    if a != b:
        raise TestFail(f"{msg}\n  expected: {b!r}\n  got:      {a!r}")


def assert_ne(a, b, msg: str) -> None:
    if a == b:
        raise TestFail(f"{msg}\n  did not expect: {b!r}")

def assert_almost_eq(a: float, b: float, eps: float = 1e-9, msg: str = "") -> None:
    if abs(a - b) > eps:
        raise TestFail(f"{msg}\n  expected ~ {b}\n  got        {a}")


def pretty_tuple(out: Tuple[float, str]) -> str:
    return f"(score={out[0]!r}, witness={out[1]!r})"


# ----------------------------
# Utilities
# ----------------------------

def lexical(metric: str, unit: str, inflect: bool) -> float:
    if metric == "aoa":
        return aoa(unit, inflect=True) if inflect else aoa(unit)
    if metric == "conc":
        return conc(unit, inflect=True) if inflect else conc(unit)
    if metric == "phon":
        return phon(unit)
    if metric == "freq":
        return freq(unit)
    raise ValueError(metric)


def find_any_known_word(metric: str, candidates: list[str], inflect: bool) -> Optional[str]:
    for w in candidates:
        if lexical(metric, w, inflect) != -1:
            return w
    return None


def find_any_known_phrase(candidates: list[str], inflect: bool) -> Optional[str]:
    for p in candidates:
        if conc(p, inflect=True) if inflect else conc(p) != -1:
            return p
    return None


# ----------------------------
# Tests
# ----------------------------

def test_tokenizer() -> None:
    s = "in the end, we win."
    toks = simple_tokenize(s)
    # Expect word-like tokens; punctuation should not be separate tokens for this tokenizer
    assert_true("in" in toks and "end" in toks and "win" in toks, f"Tokenizer seems off: {toks}")
    print("  tokenizer OK:", toks)


def test_no_eligible_words() -> None:
    # nonsense tokens likely not in any lexicon
    s = "qwertyuiop asdfghjkl zxcvbnm"
    out = score_sentence(s, metric="aoa", method="mean", skip_stopwords=False, inflect=False)
    assert_true(out[0] == -1, f"Expected -1 for no eligible words, got {pretty_tuple(out)}")
    print("  no-eligible OK:", pretty_tuple(out))


def test_methods_invariants(metric: str) -> None:
    """
    For a given metric, find at least one known word and build a sentence
    so mean/max/min are all eligible and witnesses are sensible.
    """
    candidates = [
        "dog", "cat", "house", "car", "run", "walk", "music", "water", "computer", "language",
        "child", "adult", "happy", "sad", "blue", "green",
    ]
    w1 = find_any_known_word(metric, candidates, inflect=False)
    w2 = find_any_known_word(metric, candidates[::-1], inflect=False)
    if w1 is None or w2 is None:
        print(f"  [SKIP] methods invariants for metric={metric!r}: couldn't find known words in candidates")
        return

    s = f"{w1} {w2}"
    out_mean = score_sentence(s, metric=metric, method="mean", skip_stopwords=False, inflect=False)
    out_max  = score_sentence(s, metric=metric, method="max",  skip_stopwords=False, inflect=False)
    out_min  = score_sentence(s, metric=metric, method="min",  skip_stopwords=False, inflect=False)

    assert_true(out_mean[0] != -1, f"Expected mean eligible for {metric}: {pretty_tuple(out_mean)}")
    assert_true(out_max[0] != -1 and out_max[1] in (w1, w2),
                f"Expected max witness in sentence tokens: {pretty_tuple(out_max)}")
    assert_true(out_min[0] != -1 and out_min[1] in (w1, w2),
                f"Expected min witness in sentence tokens: {pretty_tuple(out_min)}")
    assert_eq(out_mean[1], "", f"Mean witness should be empty: {pretty_tuple(out_mean)}")

    # Max should be >= Min
    assert_true(out_max[0] >= out_min[0], f"Expected max >= min for {metric}")

    print(f"  methods OK ({metric}):")
    print("    mean", pretty_tuple(out_mean))
    print("    max ", pretty_tuple(out_max))
    print("    min ", pretty_tuple(out_min))

def test_add_method_invariants() -> None:
    """
    Invariants for method='add':
      - add >= max when all scores are non-negative (common for conc/freq/phon; aoa too)
      - add == mean * N exactly by definition (within floating tolerance)
      - witness must be "" (like mean)
    We won't assume non-negativity for every lexicon, so we enforce the definitional invariant:
      add == mean * N where N is # eligible units.
    We compute N indirectly by comparing with a manually counted set of eligible tokens.
    """
    metric = "conc"  # best chance of available entries; you can also loop all metrics

    sentence = "we are the dog and the cat"  # has stopwords + content words
    tokens = simple_tokenize(sentence)

    # Count eligible units the same way score_sentence does in non-multiword mode:
    eligible_scores = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        s = conc(t)  # inflect=False in this test
        if s != -1:
            eligible_scores.append(float(s))

    out_add = score_sentence(sentence, metric=metric, method="add", skip_stopwords=True, inflect=False, multiword=False)
    out_mean = score_sentence(sentence, metric=metric, method="mean", skip_stopwords=True, inflect=False, multiword=False)

    # If none eligible, both should be -1.
    if not eligible_scores:
        assert_eq(out_add[0], -1.0, f"Expected add score -1 when no eligible units: {pretty_tuple(out_add)}")
        assert_eq(out_mean[0], -1.0, f"Expected mean score -1 when no eligible units: {pretty_tuple(out_mean)}")
        return

    N = len(eligible_scores)

    assert_true(out_add[0] != -1.0, f"Expected add score eligible: {pretty_tuple(out_add)}")
    assert_true(out_mean[0] != -1.0, f"Expected mean score eligible: {pretty_tuple(out_mean)}")
    assert_eq(out_add[1], "", f"Add witness should be empty: {pretty_tuple(out_add)}")
    assert_eq(out_mean[1], "", f"Mean witness should be empty: {pretty_tuple(out_mean)}")

    # Definition check: add == mean * N
    assert_almost_eq(out_add[0], out_mean[0] * N, msg=f"Expected add == mean*N (N={N})")

    print("  add invariants OK:", "N=", N, "add=", out_add[0], "mean=", out_mean[0])

def test_add_vs_max_when_nonnegative() -> None:
    metric = "conc"
    sentence = "dog cat house"

    tokens = simple_tokenize(sentence)
    vals = []
    for t in tokens:
        s = conc(t)
        if s != -1:
            vals.append(float(s))

    out_add = score_sentence(sentence, metric=metric, method="add", skip_stopwords=False, inflect=False, multiword=False)
    out_max = score_sentence(sentence, metric=metric, method="max", skip_stopwords=False, inflect=False, multiword=False)

    if not vals:
        # Both should be -1 if nothing is eligible
        assert_eq(out_add[0], -1.0, f"Expected add=-1 when none eligible: {pretty_tuple(out_add)}")
        assert_eq(out_max[0], -1.0, f"Expected max=-1 when none eligible: {pretty_tuple(out_max)}")
        return

    if all(v >= 0 for v in vals):
        assert_true(out_add[0] >= out_max[0], f"Expected add >= max for nonnegative scores: add={pretty_tuple(out_add)} max={pretty_tuple(out_max)}")

    print("  add vs max OK:", pretty_tuple(out_add), pretty_tuple(out_max))


def test_skip_stopwords_simple(metric: str) -> None:
    """
    In non-multiword mode (or for non-conc metrics), stopwords should be ignored entirely
    when skip_stopwords=True.
    """
    # Choose a content word that exists; sentence includes stopwords too.
    candidates = ["dog", "cat", "house", "car", "music", "water"]
    w = find_any_known_word(metric, candidates, inflect=False)
    if w is None:
        print(f"  [SKIP] skip_stopwords_simple for metric={metric!r}: couldn't find known word")
        return

    s = f"we are the {w}"  # includes stopwords
    out = score_sentence(s, metric=metric, method="max", skip_stopwords=True, inflect=False, multiword=False)

    assert_true(out[0] != -1, f"Expected eligible score for {metric}: {pretty_tuple(out)}")
    assert_true(out[1] == w, f"Expected witness to be the content word (stopwords skipped): {pretty_tuple(out)}")
    assert_true(out[1] not in STOPWORDS, f"Witness should not be a stopword here: {pretty_tuple(out)}")

    print(f"  skip_stopwords_simple OK ({metric}):", pretty_tuple(out))


def test_conc_multiword_longest_first_left_to_right() -> None:
    """
    Validate greedy behavior:
      - longest-first at each position
      - left-to-right scanning
    Because we don't know your lexicon, this test *adapts*:
      - it tries a few common phrases; if any exist, it checks matching behavior.
    """
    # Phrases with nested options (longer and shorter)
    phrase_sets = [
        ("new york city", ["new york", "york city"]),
        ("in the end", ["the end", "in the"]),
        ("at the end", ["the end"]),
        ("a lot of", ["a lot"]),
    ]

    for long_phrase, shorter in phrase_sets:
        long_exists = conc(long_phrase) != -1
        if not long_exists:
            continue

        # Construct sentence where long_phrase begins at token 0
        s = f"{long_phrase} today"
        out = score_sentence(
            s, metric="conc", multiword=True, method="max",
            skip_stopwords=False, inflect=False
        )
        # If long phrase exists, greedy should match it, not shorter subphrases.
        assert_eq(out[1], long_phrase, f"Expected longest-first match: {pretty_tuple(out)}")
        print("  conc multiword longest-first OK:", pretty_tuple(out))
        return

    print("  [SKIP] conc multiword longest-first: none of the candidate phrases exist in conc table")


def test_conc_multiword_stopword_rule() -> None:
    """
    Tests your special rule:
      - stopwords can be part of an expression and can begin one
      - BUT if skip_stopwords=True and no expression found starting at stopword, ignore it alone
      - AND (if you added the fix) reject single-token stopword matches when skip_stopwords=True
        while still allowing longer expressions starting with stopwords.
    """
    # 1) If "in the end" exists, it should be allowed even with skip_stopwords=True
    phrase = "in the end"
    if conc(phrase) != -1:
        s = f"{phrase} we win"
        out = score_sentence(s, metric="conc", multiword=True, method="max", skip_stopwords=True, inflect=False)
        assert_true(out[0] != -1, f"Expected eligible: {pretty_tuple(out)}")
        # witness should be a phrase or some other non-stopword, but at least it must not be a *single* stopword
        assert_true(not (out[1] in STOPWORDS and " " not in out[1]),
                    f"Witness unexpectedly a single stopword with skip_stopwords=True: {pretty_tuple(out)}")
        print("  conc stopword-phrase OK:", pretty_tuple(out))
    else:
        print("  [SKIP] conc stopword-phrase: 'in the end' not in conc table")

    # 2) Lone stopword should not be scored if no expression starts there
    # We'll craft sentence: stopword + nonsense, so no phrase should exist, and stopword should be ignored.
    s2 = "we qwertyuiop"
    out2 = score_sentence(s2, metric="conc", multiword=True, method="max", skip_stopwords=True, inflect=False)
    # It might still be -1 (if qwertyuiop not found), but witness must not be 'we'
    assert_true(out2[1] != "we", f"Lone stopword should be ignored when no expression match: {pretty_tuple(out2)}")
    print("  conc stopword-lone-ignore OK:", pretty_tuple(out2))

    # 3) If conc("we") exists, your earlier bug would return it.
    # With the *added fix* (reject single-token stopword matches), it should NOT return 'we'.
    we_score = conc("we")
    if we_score != -1:
        out3 = score_sentence("we win", metric="conc", multiword=True, method="max", skip_stopwords=True, inflect=False)
        assert_true(not (out3[1] == "we"),
                    f"Expected 'we' to be rejected as single-token stopword match after fix: {pretty_tuple(out3)}")
        print("  conc reject-single-stopword-match OK:", pretty_tuple(out3))
    else:
        print("  [INFO] conc('we') not found in your table; can't reproduce the earlier symptom here")


def test_inflect_flag() -> None:
    """
    Doesn't enforce numeric differences; just checks calling works and doesn't crash.
    """
    s = "running cats"
    out1 = score_sentence(s, metric="aoa", method="mean", skip_stopwords=False, inflect=False)
    out2 = score_sentence(s, metric="aoa", method="mean", skip_stopwords=False, inflect=True)
    # Either may be -1 depending on your table, but should run.
    assert_true(isinstance(out1[0], float) and isinstance(out2[0], float), "Inflect test didn't return floats")
    print("  inflect flag OK:", pretty_tuple(out1), pretty_tuple(out2))


# ----------------------------
# Runner
# ----------------------------

def main() -> None:
    tests = [
        ("tokenizer", test_tokenizer),
        ("no_eligible", test_no_eligible_words),
        ("methods_aoa", lambda: test_methods_invariants("aoa")),
        ("methods_conc", lambda: test_methods_invariants("conc")),
        ("methods_phon", lambda: test_methods_invariants("phon")),
        ("methods_freq", lambda: test_methods_invariants("freq")),
        ("skip_stopwords_aoa", lambda: test_skip_stopwords_simple("aoa")),
        ("skip_stopwords_conc_simple", lambda: test_skip_stopwords_simple("conc")),
        ("skip_stopwords_phon", lambda: test_skip_stopwords_simple("phon")),
        ("skip_stopwords_freq", lambda: test_skip_stopwords_simple("freq")),
        ("add_invariants", test_add_method_invariants),
        ("add_vs_max_nonnegative", test_add_vs_max_when_nonnegative),
        ("conc_multiword_greedy", test_conc_multiword_longest_first_left_to_right),
        ("conc_multiword_stopword_rule", test_conc_multiword_stopword_rule),
        ("inflect", test_inflect_flag),
    ]

    passed = 0
    skipped_or_info = 0

    print("Running sentence_scoring tests...\n")

    for name, fn in tests:
        try:
            print(f"[TEST] {name}")
            fn()
            passed += 1
        except TestFail as e:
            print(f"❌ FAIL: {name}\n{e}\n")
            raise
        except Exception as e:
            print(f"❌ ERROR: {name}\n{type(e).__name__}: {e}\n")
            raise
        finally:
            print()

    print(f"✅ All tests completed. Passed: {passed}/{len(tests)}")


if __name__ == "__main__":
    main()
