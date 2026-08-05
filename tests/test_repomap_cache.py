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


class TestRepoMapCacheKey:
    """The cache key must include max_tokens so a different limit is honored.

    Regression: a cache hit returned the render from the previous max_tokens,
    silently ignoring the requested limit (R3 F1).
    """

    def test_cache_key_includes_max_tokens(self, tmp_path):
        # Enough functions that a small token budget truncates the render.
        (tmp_path / "mod.py").write_text(
            "def f1(): pass\ndef f2(): pass\ndef f3(): pass\ndef f4(): pass\n"
        )
        repo_map = RepoMap(tmp_path)

        big = repo_map.get_repo_map(max_tokens=2000)
        small = repo_map.get_repo_map(max_tokens=1)

        # A 1-token budget must truncate; the two renders must differ.
        assert big != small, "different max_tokens returned identical render (cache ignored limit)"
        assert len(small) < len(big)

    def test_cache_key_stable_for_same_max_tokens(self, tmp_path):
        (tmp_path / "mod.py").write_text("def fn(): pass\n")
        repo_map = RepoMap(tmp_path)

        first = repo_map.get_repo_map(max_tokens=2000)
        second = repo_map.get_repo_map(max_tokens=2000)
        assert first == second


class TestTagExtractorStaleCache:
    """TagExtractor cache must not serve stale tags when (mtime, size) collide.

    Regression: the cache was keyed on (path, mtime_ns, size) with no content
    hash. A same-size edit in the same mtime second (realistic on NFS / Docker
    bind-mounts with second-resolution mtime) returned the old tags (R3 F2).
    """

    def test_same_size_same_mtime_edit_invalidates_cache(self, tmp_path):
        import os

        from supercoder.repomap.tag_extractor import TagExtractor

        f = tmp_path / "mod.py"
        f.write_text("def alpha(): pass\n")  # 18 bytes
        extractor = TagExtractor()

        # Capture the mtime the cache will key on (the value at first extract).
        fill_mtime_ns = f.stat().st_mtime_ns
        first = extractor.extract(str(f))
        assert [t.name for t in first] == ["alpha"]

        # Same-size edit: 'alpha' -> 'bravo' (both 18 bytes).
        f.write_text("def bravo(): pass\n")
        # Force the EXACT same mtime_ns the cache key used, simulating a
        # second-resolution filesystem where two same-size edits in the same
        # second are indistinguishable by (mtime, size) alone.
        os.utime(f, ns=(fill_mtime_ns, fill_mtime_ns))
        assert f.stat().st_size == 18

        second = extractor.extract(str(f))
        names = [t.name for t in second]
        assert "bravo" in names, f"stale tags served: {names}"
        assert "alpha" not in names

    def test_unchanged_file_cache_hit_skips_extract(self, tmp_path):
        from supercoder.repomap.tag_extractor import TagExtractor

        f = tmp_path / "mod.py"
        f.write_text("def fn(): pass\n")
        extractor = TagExtractor()

        calls = {"n": 0}
        original = extractor._do_extract

        def counting_extract(file_path):
            calls["n"] += 1
            return original(file_path)

        extractor._do_extract = counting_extract

        extractor.extract(str(f))
        extractor.extract(str(f))

        # Second call is a cache hit: no re-parse.
        assert calls["n"] == 1
