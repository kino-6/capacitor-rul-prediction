"""Property-based tests for PathUtils."""

import os
from pathlib import Path
from hypothesis import given, strategies as st, settings

from nasa_pcoe_eda.utils.paths import PathUtils


class TestPathUtilsProperties:
    """Property-based tests for PathUtils."""

    # Feature: nasa-pcoe-eda, Property 1: パス処理のOS独立性
    @given(st.data())
    @settings(max_examples=100)
    def test_path_processing_os_independence(self, data):
        """
        任意のファイルパス文字列に対して、システムのパス処理関数は、Windows形式（バックスラッシュ）と
        Unix形式（スラッシュ）の両方で同じ論理的な結果を生成する
        
        Property 1: Path processing OS independence
        For any file path string, the system's path processing functions should generate 
        the same logical result for both Windows format (backslashes) and Unix format (forward slashes).
        """
        # Generate path components
        num_components = data.draw(st.integers(min_value=1, max_value=5))
        path_components = []
        
        for _ in range(num_components):
            # Generate valid path component names (avoiding problematic characters)
            component = data.draw(st.text(
                alphabet=st.characters(
                    whitelist_categories=('Lu', 'Ll', 'Nd'),  # Letters and digits
                    whitelist_characters='_-.'
                ),
                min_size=1,
                max_size=10
            ).filter(lambda x: x not in ['', '.', '..'] and not x.startswith('.')))
            path_components.append(component)
        
        # Create Windows-style path (with backslashes)
        windows_path = '\\'.join(path_components)
        
        # Create Unix-style path (with forward slashes)
        unix_path = '/'.join(path_components)
        
        # Test normalize_path function
        normalized_windows = PathUtils.normalize_path(windows_path)
        normalized_unix = PathUtils.normalize_path(unix_path)
        
        # Both should resolve to the same logical path structure
        # Compare the parts (components) rather than the string representation
        # since the actual separator will be OS-dependent
        assert normalized_windows.parts == normalized_unix.parts, (
            f"Path normalization should produce the same logical structure for both "
            f"Windows path '{windows_path}' and Unix path '{unix_path}'. "
            f"Got Windows parts: {normalized_windows.parts}, Unix parts: {normalized_unix.parts}. "
            f"This violates the OS independence property."
        )
        
        # Test join_paths function with different input formats
        if len(path_components) >= 2:
            # Test that join_paths works with both string and Path inputs
            string_components = [str(comp) for comp in path_components]
            path_objects = [Path(comp) for comp in path_components]
            
            # All should produce the same result
            joined_from_strings = PathUtils.join_paths(*string_components)
            joined_from_paths = PathUtils.join_paths(*path_objects)
            direct_join = PathUtils.join_paths(*path_components)
            
            # The logical structure should be the same
            assert joined_from_strings.parts == direct_join.parts, (
                f"join_paths should handle string inputs consistently. "
                f"Direct join parts: {direct_join.parts}, "
                f"String join parts: {joined_from_strings.parts}. "
                f"This violates the OS independence property."
            )
            
            assert joined_from_paths.parts == direct_join.parts, (
                f"join_paths should handle Path object inputs consistently. "
                f"Direct join parts: {direct_join.parts}, "
                f"Path object join parts: {joined_from_paths.parts}. "
                f"This violates the OS independence property."
            )
        
        # Test that file operations work consistently with normalized paths
        if path_components:
            # Normalize both path formats first
            normalized_windows = PathUtils.normalize_path(windows_path)
            normalized_unix = PathUtils.normalize_path(unix_path)
            
            # Test get_filename with normalized paths
            windows_filename = PathUtils.get_filename(normalized_windows)
            unix_filename = PathUtils.get_filename(normalized_unix)
            
            assert windows_filename == unix_filename, (
                f"get_filename should return the same result for both normalized path formats. "
                f"Windows result: '{windows_filename}', Unix result: '{unix_filename}'. "
                f"This violates the OS independence property."
            )
            
            # Test get_file_extension with normalized paths
            windows_ext = PathUtils.get_file_extension(normalized_windows)
            unix_ext = PathUtils.get_file_extension(normalized_unix)
            
            assert windows_ext == unix_ext, (
                f"get_file_extension should return the same result for both normalized path formats. "
                f"Windows result: '{windows_ext}', Unix result: '{unix_ext}'. "
                f"This violates the OS independence property."
            )