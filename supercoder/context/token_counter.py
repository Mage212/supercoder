"""Token counting utilities."""

from functools import lru_cache


class TokenCounter:
    """Count tokens in text, with optional tiktoken support."""
    
    def __init__(self, use_tiktoken: bool = True, model: str = "gpt-4"):
        self.encoder = None
        self.model = model
        
        if use_tiktoken:
            try:
                import tiktoken
                # Try to get encoding for specific model
                try:
                    self.encoder = tiktoken.encoding_for_model(model)
                except KeyError:
                    # Fallback to cl100k_base (GPT-4 encoding)
                    self.encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                # tiktoken not installed, will use estimation
                pass
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        
        if self.encoder:
            return len(self.encoder.encode(text))
        
        # Fallback estimation: ~4 chars per token for English/code
        return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count without tiktoken."""
        # More accurate estimation for code:
        # - Count words (split on whitespace and punctuation)
        # - Account for code-specific patterns
        
        # Basic word count
        words = len(text.split())
        
        # Character-based estimate
        chars = len(text)
        char_estimate = chars // 4
        
        # Use the higher of the two estimates
        return max(words, char_estimate)
    
    def count_messages(self, messages: list) -> int:
        """Count tokens in a list of messages."""
        total = 0
        for msg in messages:
            # Add overhead per message (~4 tokens for role, formatting)
            total += 4
            if hasattr(msg, 'content'):
                total += self.count(msg.content)
            elif isinstance(msg, dict):
                total += self.count(msg.get('content', ''))
        return total
    
    @property
    def has_accurate_counting(self) -> bool:
        """Check if we have tiktoken for accurate counting."""
        return self.encoder is not None


# Global instance for convenience
_default_counter = None


def get_token_counter(use_tiktoken: bool = True) -> TokenCounter:
    """Get or create a global token counter."""
    global _default_counter
    if _default_counter is None:
        _default_counter = TokenCounter(use_tiktoken=use_tiktoken)
    return _default_counter


def count_tokens(text: str) -> int:
    """Convenience function to count tokens."""
    return get_token_counter().count(text)
