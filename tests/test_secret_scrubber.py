"""Tests for the secret scrubber (Phase 3C, backlog B-036)."""

from supercoder.utils.secret_scrubber import MASK, scrub_secrets

_GITHUB_PAT = "ghp_" + "a" * 36
_OPENAI_KEY = "sk-" + "b" * 36


class TestScrubPatterns:
    def test_masks_openai_key(self):
        scrubbed = scrub_secrets(f"key is {_OPENAI_KEY} end")
        assert _OPENAI_KEY not in scrubbed
        assert MASK in scrubbed

    def test_masks_openrouter_key(self):
        scrubbed = scrub_secrets("sk-or-v1-abcdef1234567890")
        assert "sk-or-v1-abcdef1234567890" not in scrubbed

    def test_masks_github_pat(self):
        scrubbed = scrub_secrets(f"token={_GITHUB_PAT}")
        assert _GITHUB_PAT not in scrubbed

    def test_masks_aws_access_key(self):
        scrubbed = scrub_secrets("AKIAIOSFODNN7EXAMPLE")
        assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed

    def test_masks_aws_secret_access_key(self):
        secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        scrubbed = scrub_secrets(f'aws_secret_access_key = "{secret}"')
        assert secret not in scrubbed
        assert MASK in scrubbed

    def test_masks_pem_block(self):
        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
        scrubbed = scrub_secrets(f"cert:\n{pem}\ndone")
        assert "MIIEpAIBAAKCAQEA" not in scrubbed
        assert MASK in scrubbed

    def test_masks_bearer_header(self):
        scrubbed = scrub_secrets("Authorization: Bearer abc.def.ghi-jkl_mno")
        assert "abc.def.ghi-jkl_mno" not in scrubbed

    def test_masks_generic_assignment(self):
        scrubbed = scrub_secrets('api_key: "secretvalue123"')
        assert "secretvalue123" not in scrubbed
        assert MASK in scrubbed

    def test_masks_generic_with_password(self):
        scrubbed = scrub_secrets("password=hunter2pass")
        assert "hunter2pass" not in scrubbed


class TestScrubNoFalsePositives:
    def test_short_sk_not_matched(self):
        # "sk-learn" has only 5 chars after sk-; pattern requires >= 20.
        assert scrub_secrets("import sk-learn stuff") == "import sk-learn stuff"

    def test_normal_text_unchanged(self):
        text = "The function compute_hash returns a digest."
        assert scrub_secrets(text) == text

    def test_short_generic_value_not_matched(self):
        # Generic pattern requires >= 8 chars in the value.
        assert scrub_secrets("token: abc") == "token: abc"


class TestScrubRecursion:
    def test_recursive_dict(self):
        data = {"config": {"nested_key": _OPENAI_KEY}, "plain": "keep"}
        scrubbed = scrub_secrets(data)
        assert _OPENAI_KEY not in scrubbed["config"]["nested_key"]
        assert scrubbed["plain"] == "keep"

    def test_recursive_list(self):
        data = [_OPENAI_KEY, "normal", {"deep": _GITHUB_PAT}]
        scrubbed = scrub_secrets(data)
        assert _OPENAI_KEY not in scrubbed[0]
        assert scrubbed[1] == "normal"
        assert _GITHUB_PAT not in scrubbed[2]["deep"]

    def test_preserves_non_string_types(self):
        data = {"count": 42, "ratio": 1.5, "active": True, "nothing": None}
        assert scrub_secrets(data) == data

    def test_preserves_dict_keys(self):
        scrubbed = scrub_secrets({"api_key": _OPENAI_KEY})
        # Key preserved, value masked.
        assert "api_key" in scrubbed
        assert scrubbed["api_key"] != _OPENAI_KEY

    def test_empty_and_none(self):
        assert scrub_secrets("") == ""
        assert scrub_secrets(None) is None
