"""Tests for RepoMap determinism and caching (epoch-memory-arch #1).

These tests verify the cache-friendly properties required for prefix-cache
stability with local LLM backends:

- Deterministic file selection (sorted) and rendering order across calls,
  independent of filesystem inode/directory order.
- Content-hash cache: identical files return identical bytes without
  re-running extraction; changed files trigger a cache miss.
- A sidecar metadata file records the cache key for human inspection.
"""

from supercoder.repomap import RepoMap


class TestRepoMapDeterminism:
    """RepoMap output must be byte-identical for identical inputs."""

    def test_identical_content_across_calls(self, tmp_path):
        """Repeated calls on unchanged files return byte-identical output."""
        (tmp_path / "a.py").write_text("def alpha(): pass\n")
        (tmp_path / "b.py").write_text("def beta(): pass\n")

        repo_map = RepoMap(tmp_path)
        first = repo_map.get_repo_map(max_tokens=2048)
        second = repo_map.get_repo_map(max_tokens=2048)

        assert first == second

    def test_file_order_independent(self, tmp_path):
        """Creation order on disk must not affect rendered output.

        rglob order depends on directory inode order and is not stable across
        runs; the map must sort selected files before rendering.
        """
        # Create files in non-alphabetical order
        (tmp_path / "zeta.py").write_text("def z(): pass\n")
        (tmp_path / "alpha.py").write_text("def a(): pass\n")
        (tmp_path / "mid.py").write_text("def m(): pass\n")

        repo_map = RepoMap(tmp_path)
        content = repo_map.get_repo_map(max_tokens=2048)

        # alpha should appear before mid, mid before zeta
        pos_alpha = content.find("alpha.py")
        pos_mid = content.find("mid.py")
        pos_zeta = content.find("zeta.py")

        assert pos_alpha != -1
        assert pos_mid != -1
        assert pos_zeta != -1
        assert pos_alpha < pos_mid < pos_zeta

    def test_get_files_sorted(self, tmp_path):
        """_get_files returns sorted paths for deterministic selection."""
        (tmp_path / "c.py").write_text("def c(): pass\n")
        (tmp_path / "a.py").write_text("def a(): pass\n")
        (tmp_path / "b.py").write_text("def b(): pass\n")

        repo_map = RepoMap(tmp_path)
        files = repo_map._get_files()

        rel_names = [f.name for f in files]
        assert rel_names == sorted(rel_names)


class TestRepoMapCache:
    """Content-hash cache avoids redundant work for unchanged files."""

    def test_cache_hit_returns_identical_bytes(self, tmp_path):
        """Identical file set + content returns the cached string unchanged."""
        (tmp_path / "mod.py").write_text("def fn(): pass\n")

        repo_map = RepoMap(tmp_path)
        first = repo_map.get_repo_map(max_tokens=2048)

        # Mutate the extractor to prove the cache short-circuits: if the cache
        # hit, extract() must not be called again for the same file.
        call_count = {"n": 0}
        original_extract = repo_map.extractor.extract

        def counting_extract(file_path):
            call_count["n"] += 1
            return original_extract(file_path)

        repo_map.extractor.extract = counting_extract

        second = repo_map.get_repo_map(max_tokens=2048)

        assert first == second
        # Cache hit: no extraction needed on the second call
        assert call_count["n"] == 0

    def test_cache_miss_on_content_change(self, tmp_path):
        """Changing a file's content invalidates the cache and regenerates."""
        f = tmp_path / "mod.py"
        f.write_text("def original(): pass\n")

        repo_map = RepoMap(tmp_path)
        first = repo_map.get_repo_map(max_tokens=2048)
        assert "original" in first

        # Change content (keep size/mtime may collide, so sleep to bump mtime)
        import time

        time.sleep(0.01)
        f.write_text("def renamed(): pass\n")

        second = repo_map.get_repo_map(max_tokens=2048)

        assert "renamed" in second
        assert "original" not in second
        assert first != second

    def test_meta_sidecar_written_with_cache_key(self, tmp_path):
        """A sidecar repo_map.meta.json records the cache key and freshness."""
        (tmp_path / "mod.py").write_text("def fn(): pass\n")

        repo_map = RepoMap(tmp_path)
        repo_map.get_repo_map(max_tokens=2048)

        import json

        meta_path = repo_map.storage_dir / "repo_map.meta.json"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert "cache_key" in meta
        assert "generated_at" in meta
        assert "file_count" in meta
        assert meta["file_count"] == 1

    def test_repo_map_txt_written_only_on_cache_miss(self, tmp_path):
        """repo_map.txt is rewritten only when the map actually changes."""
        (tmp_path / "mod.py").write_text("def fn(): pass\n")

        repo_map = RepoMap(tmp_path)
        repo_map.get_repo_map(max_tokens=2048)

        map_path = repo_map.storage_dir / "repo_map.txt"
        assert map_path.exists()
        first_mtime = map_path.stat().st_mtime_ns

        # Second call is a cache hit: file must not be rewritten
        repo_map.get_repo_map(max_tokens=2048)
        second_mtime = map_path.stat().st_mtime_ns

        assert second_mtime == first_mtime


class TestRepoMapNoGraphLeak:
    """The dead nx.MultiDiGraph was removed; nodes must not accumulate."""

    def test_no_graph_attribute(self, tmp_path):
        """RepoMap no longer carries a networkx graph (dead code removed)."""
        repo_map = RepoMap(tmp_path)
        assert not hasattr(repo_map, "graph")
