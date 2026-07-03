"""Tests for :mod:`bgkit.data.opaque_ids` (BIP-39 opaque id primitives)."""

from __future__ import annotations

from bgkit.data.opaque_ids import (
    BIP39_ENGLISH,
    bip39_id,
    single_token_bip39_wordlist,
)


def test_wordlist_is_canonical_bip39():
    assert len(BIP39_ENGLISH) == 2048
    assert len(set(BIP39_ENGLISH)) == 2048
    assert BIP39_ENGLISH[0] == "abandon"
    assert BIP39_ENGLISH[-1] == "zoo"


def test_bip39_id_deterministic():
    assert bip39_id("hello") == bip39_id("hello")
    assert bip39_id("a-commit-sha") == bip39_id("a-commit-sha")


def test_bip39_id_is_word_based_not_positional():
    out = bip39_id("deadbeef")
    parts = out.split("-")
    assert len(parts) == 2
    for p in parts:
        assert p in BIP39_ENGLISH
    # No positional markers anywhere.
    assert "cm_" not in out and "c4_" not in out and "c16_" not in out


def test_bip39_id_n_words():
    assert len(bip39_id("x", n_words=1).split("-")) == 1
    assert len(bip39_id("x", n_words=3).split("-")) == 3
    assert bip39_id("x", n_words=1) in BIP39_ENGLISH


def test_bip39_id_collision_free_on_toy_set():
    ids = {bip39_id(f"sha-{i}") for i in range(2000)}
    assert len(ids) == 2000  # no collisions over 2048**2 space


def test_bip39_id_distinct_inputs_distinct_ids():
    assert bip39_id("abc") != bip39_id("abd")


def test_bip39_id_custom_wordlist():
    wl = ["foo", "bar", "baz", "qux"]
    out = bip39_id("anything", n_words=2, wordlist=wl)
    for p in out.split("-"):
        assert p in wl


class _FakeTokenizer:
    """Encodes a word to one token iff it is in ``single``."""

    def __init__(self, single: set[str], leading_space: bool = True):
        self._single = single
        self._leading_space = leading_space

    def encode(self, text, add_special_tokens=False):
        w = text[1:] if (self._leading_space and text.startswith(" ")) else text
        return [1] if w in self._single else [1, 2]


def test_single_token_filter_intersection():
    words = list(BIP39_ENGLISH[:50])
    tok_a = _FakeTokenizer(set(words[:40]))
    tok_b = _FakeTokenizer(set(words[10:50]))
    stats: dict = {}
    result = single_token_bip39_wordlist(
        [tok_a, tok_b], wordlist=words, stats=stats,
    )
    # intersection = words[10:40] = 30 words, preserving order
    assert result == words[10:40]
    assert stats["per_tokenizer"][0]["survivors"] == 40
    assert stats["per_tokenizer"][1]["survivors"] == 40
    assert stats["intersection"] == 30
    assert stats["degraded"] is False


def test_single_token_filter_degrades_on_no_tokenizers():
    stats: dict = {}
    result = single_token_bip39_wordlist([], wordlist=list(BIP39_ENGLISH[:20]), stats=stats)
    assert result == list(BIP39_ENGLISH[:20])
    assert stats["degraded"] is True


def test_single_token_filter_degrades_on_broken_tokenizer():
    class _Broken:
        def encode(self, *a, **k):
            raise RuntimeError("boom")

    words = list(BIP39_ENGLISH[:20])
    stats: dict = {}
    result = single_token_bip39_wordlist([_Broken()], wordlist=words, stats=stats)
    assert result == words
    assert stats["degraded"] is True
