"""Artifact tags are labels, not identifiers: Unicode letters, marks and digits, stored NFC.

One rule (``normalize_tag``) is read by the store and by the MCP argument gate,
so what the tool accepts the store accepts, in the same spelling. These tests pin
the rule from both sides and the equalities between them.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from kiro_crew.artifact_store import rules
from kiro_crew.artifacts import ArtifactStore, ArtifactValidationError, normalize_tag
from kiro_crew.validation import (
    ARTIFACT_LIST_SCHEMA,
    ARTIFACT_SAVE_SCHEMA,
    ARTIFACT_UPDATE_SCHEMA,
    ValidationError,
    strip_hidden_unicode,
    validate_tool_args,
)

NFD_CAFE = "cafe\u0301"  # e + combining acute: two code points
NFC_CAFE = "caf\u00e9"  # precomposed e-acute: one code point


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(root=tmp_path / "artifacts")


class TestNormalizeTag:
    @pytest.mark.parametrize(
        "tag",
        [
            "\u58f2\u4e0a",  # 売上 (CJK ideographs)
            NFC_CAFE,
            "M\u00fcnchen",
            "\u0939\u093f\u0928\u094d\u0926\u0940",  # हिन्दी: vowel signs are marks NFC keeps apart
            "\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac",  # Ελληνικά
            "\u0440\u0443\u0441\u0441\u043a\u0438\u0439",  # русский
            "\u0627\u0644\u0639\u0631\u0628\u064a\u0629",  # العربية
            "\u65e5\u672c\u8a9e2026",  # letters and digits mixed
            "\u0661\u0662\u0663",  # Arabic-Indic digits: N* is admitted, not just 0-9
        ],
    )
    def test_letters_marks_and_digits_of_any_script_are_admitted(self, tag: str) -> None:
        assert normalize_tag(tag) == tag

    def test_the_stored_spelling_is_nfc(self) -> None:
        assert normalize_tag(NFD_CAFE) == NFC_CAFE
        assert normalize_tag(NFC_CAFE) == NFC_CAFE

    @pytest.mark.parametrize(
        "tag",
        ["release", "v1.2", "ns:name", "snake_case", "kebab-case", "9lives", "A.B_C:d-e", "a" * 64],
    )
    def test_existing_ascii_tags_pass_unchanged(self, tag: str) -> None:
        assert normalize_tag(tag) == tag

    @pytest.mark.parametrize(
        "tag",
        [
            "cr\n",  # the ``$`` anchor's before-newline match must stay closed
            "a" * 63 + "\n",  # 64 code points, so the newline is what is refused
            "tab\tx",
            "nul\x00x",
            "bad tag with spaces",
            "a\u00a0b",  # no-break space is whitespace too
            "a\u200bb",  # zero-width space (format)
            "a\u202eb",  # right-to-left override (format)
            "a\ufeffb",  # byte-order mark (format)
        ],
    )
    def test_control_format_and_whitespace_characters_are_rejected(self, tag: str) -> None:
        with pytest.raises(ValueError, match="is not allowed in a tag"):
            normalize_tag(tag)

    @pytest.mark.parametrize("tag", ["a/b", "a'b", "a,b", "a\U0001f642", "a\u20ac", "a+b", "a@b"])
    def test_symbols_and_other_punctuation_are_rejected(self, tag: str) -> None:
        with pytest.raises(ValueError, match="is not allowed in a tag"):
            normalize_tag(tag)

    @pytest.mark.parametrize(
        "ch",
        [
            "\u034f",  # COMBINING GRAPHEME JOINER (Mn)
            "\u115f",  # HANGUL CHOSEONG FILLER (Lo)
            "\u1160",  # HANGUL JUNGSEONG FILLER (Lo)
            "\u17b4",  # KHMER VOWEL INHERENT AQ (Mn)
            "\u180b",  # MONGOLIAN FREE VARIATION SELECTOR ONE (Mn)
            "\u180f",  # MONGOLIAN FREE VARIATION SELECTOR FOUR (Mn)
            "\u3164",  # HANGUL FILLER (Lo)
            "\ufe00",  # VARIATION SELECTOR-1 (Mn)
            "\ufe0f",  # VARIATION SELECTOR-16 (Mn): the emoji-presentation selector
            "\uffa0",  # HALFWIDTH HANGUL FILLER (Lo)
            "\U000e0100",  # VARIATION SELECTOR-17 (Mn)
            "\U000e01ef",  # VARIATION SELECTOR-256 (Mn)
        ],
    )
    def test_invisible_letters_and_marks_are_rejected(self, ch: str) -> None:
        # Every one of these is a letter or a mark by general category, and every
        # one renders as nothing (Default_Ignorable_Code_Point). Admitted, it
        # would make ``ops`` and ``ops<VS16>`` two stored tags with one look, and
        # let ``AKIA<VS16>IOSFODNN7EXAMPLE`` carry a key id past a redactor that
        # matches ``AKIA[A-Z0-9]{16}``, invisible to the reader it is shown to.
        assert unicodedata.category(ch)[0] in ("L", "M")
        with pytest.raises(ValueError, match=r"renders as nothing"):
            normalize_tag("ops" + ch)
        with pytest.raises(ValueError, match=r"U\+%04X" % ord(ch)):
            normalize_tag("ops" + ch + "x")

    def test_the_invisible_table_is_the_published_property(self) -> None:
        # Default_Ignorable_Code_Point, Unicode 15.0 DerivedCoreProperties.txt:
        # 4,174 code points. Pinning the count catches an edit that drops a range.
        assert len(rules._DEFAULT_IGNORABLE) == 4174
        assert {0x00AD, 0x034F, 0x200B, 0x3164, 0xFE0F, 0xFEFF, 0xE0001, 0xE01EF} <= set(
            rules._DEFAULT_IGNORABLE
        )
        # Not in the property, and not to be: the shaping joiners ZWNJ/ZWJ are
        # format characters the rule refuses on category; a visible accent is kept.
        assert 0x0301 not in rules._DEFAULT_IGNORABLE

    @pytest.mark.parametrize(
        "tag",
        [
            "1\ufe0f\u20e3",  # keycap digit one: digit + VS16 + COMBINING ENCLOSING KEYCAP
            "1\u20e3",  # the keycap renders as an emoji without the selector too
            "a\u20dd",  # COMBINING ENCLOSING CIRCLE
            "a\u0488",  # COMBINING CYRILLIC HUNDRED THOUSANDS SIGN
            "a\u1abe",  # COMBINING PARENTHESES OVERLAY
        ],
    )
    def test_enclosing_marks_are_rejected(self, tag: str) -> None:
        # ``Me`` marks draw a shape around their base: that is a symbol or an
        # emoji, not a letter, so they are refused with the symbols.
        with pytest.raises(ValueError, match="is not allowed in a tag"):
            normalize_tag(tag)

    @pytest.mark.parametrize(
        "tag",
        [
            "\u0915\u093e\u092e",  # काम: Devanagari vowel sign AA is a spacing mark (Mc)
            "\u0e19\u0e49\u0e33",  # น้ำ: Thai tone mark MAI THO is a nonspacing mark (Mn)
            "\u0645\u064e\u0631\u0652\u062d\u064e\u0628\u0627",  # مَرْحَبا: harakat (Mn)
            "\u0dc1\u0dca\u0dbb\u0dd3",  # ශ්රී: al-lakuna (Mn) and vowel sign (Mc)
        ],
    )
    def test_the_marks_scripts_are_written_with_stay_admitted(self, tag: str) -> None:
        assert {unicodedata.category(ch) for ch in tag} & {"Mn", "Mc"}
        assert normalize_tag(tag) == tag

    @pytest.mark.parametrize("tag", ["-bad", ":x", ".x", "_x", "\u0301x", "\U0001f642", "\u20ac1"])
    def test_first_character_must_be_a_letter_or_digit(self, tag: str) -> None:
        with pytest.raises(ValueError, match="must start with a letter or digit"):
            normalize_tag(tag)

    def test_empty_tag_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_tag("")

    def test_cap_counts_code_points_after_nfc(self) -> None:
        assert normalize_tag("\u58f2" * rules.MAX_TAG_LEN) == "\u58f2" * rules.MAX_TAG_LEN
        with pytest.raises(ValueError, match="at most 64 characters"):
            normalize_tag("\u58f2" * (rules.MAX_TAG_LEN + 1))
        # 128 code points on the way in, 64 once composed: the cap reads the stored form.
        assert normalize_tag("e\u0301" * rules.MAX_TAG_LEN) == "\u00e9" * rules.MAX_TAG_LEN
        with pytest.raises(ValueError, match="at most 64 characters"):
            normalize_tag("a" * (rules.MAX_TAG_LEN + 1))

    def test_rejection_names_the_offending_code_point(self) -> None:
        with pytest.raises(ValueError, match=r"U\+000A"):
            normalize_tag("cr\n")


class TestStoreTags:
    @pytest.mark.parametrize("tag", ["\u58f2\u4e0a", NFC_CAFE, "M\u00fcnchen"])
    def test_non_ascii_tags_are_accepted_and_persisted(
        self, store: ArtifactStore, tag: str
    ) -> None:
        art = store.create(name="x", content="a", tags=[tag])
        assert art.tags == [tag]
        assert store.get(art.slug).tags == [tag]

    def test_tags_are_stored_nfc(self, store: ArtifactStore) -> None:
        art = store.create(name="x", content="a", tags=[NFD_CAFE])
        assert art.tags == [NFC_CAFE]
        assert store.get(art.slug).tags == [NFC_CAFE]

    def test_equivalent_spellings_dedupe_to_one_tag(self, store: ArtifactStore) -> None:
        art = store.create(name="x", content="a", tags=[NFC_CAFE, NFD_CAFE, "ops"])
        assert art.tags == [NFC_CAFE, "ops"]

    def test_update_takes_non_ascii_tags(self, store: ArtifactStore) -> None:
        art = store.create(name="x", content="a", tags=["ops"])
        updated = store.update(art.slug, tags=["\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac"])
        assert updated.tags == ["\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac"]

    @pytest.mark.parametrize(
        "tag", ["cr\n", "a\u200bb", "-bad", "a b", "\u58f2" * 65, "ops\ufe0f", "1\ufe0f\u20e3"]
    )
    def test_store_refuses_what_the_rule_refuses(self, store: ArtifactStore, tag: str) -> None:
        with pytest.raises(ArtifactValidationError, match="invalid tag"):
            store.create(name="x", content="a", tags=[tag])

    def test_a_tag_has_no_invisible_twin(self, store: ArtifactStore) -> None:
        # ``ops`` and ``ops<VS16>`` look the same on every screen. If the second
        # were a tag, ``list(tag="ops")`` would miss it; the rule refuses it, so the
        # one spelling a reader sees is the one spelling the store holds.
        art = store.create(name="x", content="a", tags=["ops"])
        with pytest.raises(ArtifactValidationError, match="renders as nothing"):
            store.create(name="y", content="a", tags=["ops\ufe0f"])
        assert [a.slug for a in store.list(tag="ops")] == [art.slug]

    def test_store_error_is_plain_english(self, store: ArtifactStore) -> None:
        with pytest.raises(ArtifactValidationError) as info:
            store.create(name="x", content="a", tags=["-bad"])
        assert str(info.value) == "invalid tag '-bad': a tag must start with a letter or digit"

    def test_list_filter_matches_the_tag_in_any_spelling(self, store: ArtifactStore) -> None:
        art = store.create(name="x", content="a", tags=[NFC_CAFE])
        assert [a.slug for a in store.list(tag=NFD_CAFE)] == [art.slug]
        assert [a.slug for a in store.list(tag=NFC_CAFE)] == [art.slug]

    def test_list_filter_that_is_not_a_tag_matches_nothing(self, store: ArtifactStore) -> None:
        store.create(name="x", content="a", tags=["ops"])
        assert store.list(tag="no way") == []


class TestMcpGateReadsTheSameRule:
    def _save_args(self, *tags: str) -> dict:
        return {"name": "x", "content": "a", "tags": list(tags)}

    def test_save_admits_non_ascii_tags(self) -> None:
        cleaned = validate_tool_args(
            self._save_args("\u58f2\u4e0a", NFD_CAFE), ARTIFACT_SAVE_SCHEMA
        )
        # The gate NFC-normalizes on the way in, so the store sees the stored spelling.
        assert cleaned["tags"] == ["\u58f2\u4e0a", NFC_CAFE]

    def test_update_admits_non_ascii_tags(self) -> None:
        args = {"slug": "x", "tags": ["M\u00fcnchen", "\u0939\u093f\u0928\u094d\u0926\u0940"]}
        cleaned = validate_tool_args(args, ARTIFACT_UPDATE_SCHEMA)
        assert cleaned["tags"] == args["tags"]

    def test_list_filter_admits_a_non_ascii_tag(self) -> None:
        assert (
            validate_tool_args({"tag": "M\u00fcnchen"}, ARTIFACT_LIST_SCHEMA)["tag"]
            == "M\u00fcnchen"
        )

    @pytest.mark.parametrize("tag", ["-bad", "a/b", "\U0001f642", "\u58f2" * 65])
    def test_save_refuses_with_the_rules_reason(self, tag: str) -> None:
        with pytest.raises(ValidationError, match="tags"):
            validate_tool_args(self._save_args(tag), ARTIFACT_SAVE_SCHEMA)

    def test_save_refuses_a_credential_hidden_behind_an_invisible_mark(self) -> None:
        # The gate's sanitizer drops controls and format characters but keeps
        # VARIATION SELECTOR-16: it is a mark (``Mn``), and emoji text needs it. So
        # the tag rule is the check that keeps ``AKIA<VS16>IOSFODNN7EXAMPLE`` out of
        # ``meta.json``, where a redactor matching ``AKIA[A-Z0-9]{16}`` would not
        # see the key id while a reader would.
        planted = "AKIA\ufe0fIOSFODNN7EXAMPLE"
        assert strip_hidden_unicode(planted) == planted
        with pytest.raises(ValidationError, match=r"item\[0\]: .*U\+FE0F.*renders as nothing"):
            validate_tool_args(self._save_args(planted), ARTIFACT_SAVE_SCHEMA)
        with pytest.raises(ValidationError, match=r"U\+FE0F"):
            validate_tool_args({"tag": planted}, ARTIFACT_LIST_SCHEMA)

    def test_save_error_carries_the_item_index_and_reason(self) -> None:
        with pytest.raises(
            ValidationError, match=r"item\[1\]: a tag must start with a letter or digit"
        ):
            validate_tool_args(self._save_args("ok", "-bad"), ARTIFACT_SAVE_SCHEMA)

    @pytest.mark.parametrize("tag", ["-bad", "a/b"])
    def test_list_filter_refuses_with_the_rules_reason(self, tag: str) -> None:
        with pytest.raises(ValidationError, match="tag"):
            validate_tool_args({"tag": tag}, ARTIFACT_LIST_SCHEMA)

    @pytest.mark.parametrize(
        "tag",
        [
            "\u58f2\u4e0a",
            NFC_CAFE,
            "M\u00fcnchen",
            "\u0939\u093f\u0928\u094d\u0926\u0940",
            "release",
            "v1.2",
            "-bad",
            "a/b",
            "\U0001f642",
            "\u58f2" * 64,
            "\u58f2" * 65,
            "ops\ufe0f",  # the sanitizer keeps a variation selector (Mn): both readers see it
            "1\ufe0f\u20e3",
            "\u3164x",  # HANGUL FILLER is a letter (Lo) to the sanitizer too
        ],
    )
    def test_store_and_gate_agree_on_visible_text(self, store: ArtifactStore, tag: str) -> None:
        # Visible text only: the gate's sanitizer strips hidden characters and edge
        # whitespace before the rule runs, so those inputs reach the two readers as
        # different strings by design.
        try:
            store.create(name="x", content="a", tags=[tag])
            store_accepts = True
        except ArtifactValidationError:
            store_accepts = False
        try:
            validate_tool_args({"name": "x", "content": "a", "tags": [tag]}, ARTIFACT_SAVE_SCHEMA)
            gate_accepts = True
        except ValidationError:
            gate_accepts = False
        assert store_accepts == gate_accepts


class TestFacade:
    def test_the_store_reads_the_rule_through_the_facade(self) -> None:
        import kiro_crew.artifacts as art_mod

        assert art_mod.normalize_tag is rules.normalize_tag
        assert art_mod.MAX_TAG_LEN == rules.MAX_TAG_LEN == 64
        assert not hasattr(rules, "_TAG_RE"), "the regex is gone: one rule, one owner"

    def test_nfc_is_the_stored_form_the_rule_promises(self) -> None:
        for tag in ["\u58f2\u4e0a", NFD_CAFE, "M\u00fcnchen"]:
            assert normalize_tag(tag) == unicodedata.normalize("NFC", tag)
