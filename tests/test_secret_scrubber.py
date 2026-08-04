"""Tests for the secret scrubber (Phase 3C, backlog B-036)."""

from supercoder.utils.secret_scrubber import MASK, scrub_secrets

_GITHUB_PAT = "ghp_" + "a" * 36
_OPENAI_KEY = "sk-" + "b" * 36
# Modern OpenAI/Anthropic key formats (default since 2024) embed a hyphen after
# the provider segment, which the legacy `sk-[A-Za-z0-9]{20,}` regex did not allow.
_OPENAI_PROJ_KEY = "sk-proj-" + "AbCdEfGhIjKlMnOpQrSt"  # bare, no key= context
_ANTHROPIC_KEY = "sk-ant-api03-" + "AbCdEfGhIjKlMnOpQrSt"


class TestScrubPatterns:
    def test_masks_openai_key(self):
        scrubbed = scrub_secrets(f"key is {_OPENAI_KEY} end")
        assert _OPENAI_KEY not in scrubbed
        assert MASK in scrubbed

    def test_masks_openrouter_key(self):
        scrubbed = scrub_secrets("sk-or-v1-abcdef1234567890")
        assert "sk-or-v1-abcdef1234567890" not in scrubbed

    def test_masks_openai_proj_key_bare(self):
        """M2: sk-proj- (default OpenAI format since 2024) in prose/traceback."""
        scrubbed = scrub_secrets(f"Error: auth failed for {_OPENAI_PROJ_KEY}")
        assert _OPENAI_PROJ_KEY not in scrubbed
        assert MASK in scrubbed

    def test_masks_anthropic_key_bare(self):
        """M2: sk-ant- format in prose/traceback."""
        scrubbed = scrub_secrets(f"Exception: key {_ANTHROPIC_KEY} rejected")
        assert _ANTHROPIC_KEY not in scrubbed
        assert MASK in scrubbed

    def test_masks_openai_proj_key_in_traceback(self):
        """Bare key appearing in a multiline traceback (no api_key= context)."""
        traceback = f"Traceback (most recent call last):\n  File x.py, line 1\n    {_OPENAI_PROJ_KEY}\nAuthError: invalid"
        scrubbed = scrub_secrets(traceback)
        assert _OPENAI_PROJ_KEY not in scrubbed

    def test_masks_google_ai_key_bare(self):
        """R2-6: Google AI / GCP key (AIza...) in bare form (no key= context)."""
        google_key = "AIzaSyA1234567890abcdefghijklmnopqrstuv"
        scrubbed = scrub_secrets(f"Error: auth failed for {google_key}")
        assert google_key not in scrubbed
        assert MASK in scrubbed

    def test_masks_replicate_token_bare(self):
        """R2-6: Replicate API token (r8_...) in bare form."""
        replicate_token = "r8_abcDEF1234567890ghijklmnop"
        scrubbed = scrub_secrets(f"token rejected: {replicate_token}")
        assert replicate_token not in scrubbed

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


class TestGenericSecretRegexFalsePositives:
    """M4 (code-review-2026-06-23): the generic key=value regex must not mask
    ordinary source-code identifiers, short placeholders, or function-call RHS.
    False positives make tool-result logs unreadable; the module docstring states
    a masked description is worse than a missed random token."""

    def test_does_not_mask_function_call_rhs(self):
        assert (
            scrub_secrets("token = get_token_from_request()") == "token = get_token_from_request()"
        )
        assert scrub_secrets("secret = generate_secret()") == "secret = generate_secret()"

    def test_does_not_mask_placeholder_literal(self):
        assert scrub_secrets('token = "placeholder"') == 'token = "placeholder"'
        assert scrub_secrets('password = "changeme"') == 'password = "changeme"'

    def test_does_not_mask_short_alnum_literal(self):
        # Short, no digit+letter mix that signals a real secret.
        assert scrub_secrets('api_key = "abcdef12"') == 'api_key = "abcdef12"'

    def test_still_masks_realistic_mixed_secret(self):
        # Mixed letters+digits, >= 12 chars — looks like a real generated key.
        assert MASK in scrub_secrets('api_key = "akLive9xK7mQ2wZ4"')
        assert MASK in scrub_secrets('token = "sk_live_42AbCdEf9999"')

    def test_still_masks_long_high_entropy(self):
        # Long enough (>=20) that it is secret-like regardless of mix.
        long_val = "abcdefghijklmnopqrstuvwxyz1234"
        assert MASK in scrub_secrets(f'secret = "{long_val}"')
