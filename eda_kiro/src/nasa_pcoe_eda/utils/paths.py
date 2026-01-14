"""OS-independent path handling utilities using pathlib"""

from pathlib import Path
from typing import Union


class PathUtils:
    """Utility class for OS-independent path operations"""

    @staticmethod
    def normalize_path(path: Union[str, Path]) -> Path:
        """
        Normalize a path to be OS-independent.
        Converts both forward slashes and backslashes to the OS-appropriate separator.
        
        Args:
            path: Path as string or Path object
            
        Returns:
            Normalized Path object
        """
        # Convert string path to use forward slashes (pathlib handles conversion)
        if isinstance(path, str):
            # Replace backslashes with forward slashes for cross-platform compatibility
            path = path.replace('\\', '/')
        return Path(path).resolve()

    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            path: Directory path as string or Path object
            
        Returns:
            Path object of the directory
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @staticmethod
    def join_paths(*parts: Union[str, Path]) -> Path:
        """
        Join multiple path components in an OS-independent way.
        
        Args:
            *parts: Path components to join
            
        Returns:
            Joined Path object
        """
        if not parts:
            return Path()
        
        result = Path(parts[0])
        for part in parts[1:]:
            result = result / part
        return result

    @staticmethod
    def get_file_extension(path: Union[str, Path]) -> str:
        """
        Get the file extension from a path.
        
        Args:
            path: File path as string or Path object
            
        Returns:
            File extension (including the dot, e.g., '.csv')
        """
        return Path(path).suffix

    @staticmethod
    def get_filename(path: Union[str, Path], with_extension: bool = True) -> str:
        """
        Get the filename from a path.
        
        Args:
            path: File path as string or Path object
            with_extension: Whether to include the file extension
            
        Returns:
            Filename as string
        """
        path_obj = Path(path)
        if with_extension:
            return path_obj.name
        return path_obj.stem

    @staticmethod
    def is_file(path: Union[str, Path]) -> bool:
        """
        Check if a path points to an existing file.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is an existing file, False otherwise
        """
        return Path(path).is_file()

    @staticmethod
    def is_dir(path: Union[str, Path]) -> bool:
        """
        Check if a path points to an existing directory.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is an existing directory, False otherwise
        """
        return Path(path).is_dir()

    @staticmethod
    def list_files(directory: Union[str, Path], pattern: str = "*") -> list[Path]:
        """
        List all files in a directory matching a pattern.
        
        Args:
            directory: Directory path
            pattern: Glob pattern for matching files (default: "*")
            
        Returns:
            List of Path objects for matching files
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return []
        return [f for f in dir_path.glob(pattern) if f.is_file()]

    @staticmethod
    def get_parent_dir(path: Union[str, Path]) -> Path:
        """
        Get the parent directory of a path.
        
        Args:
            path: Path as string or Path object
            
        Returns:
            Parent directory as Path object
        """
        return Path(path).parent

    @staticmethod
    def relative_to(path: Union[str, Path], base: Union[str, Path]) -> Path:
        """
        Get a path relative to a base path.
        
        Args:
            path: Path to make relative
            base: Base path
            
        Returns:
            Relative path
        """
        return Path(path).relative_to(Path(base))
