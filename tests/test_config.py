"""
Tests for application configuration.

These tests ensure that required settings are properly defined
to prevent runtime errors from missing configuration.
"""
import pytest


class TestSettings:
    """Test application settings configuration."""

    def test_settings_has_storage_path(self):
        """Ensure storage_path is defined in Settings.

        This prevents the 'Settings' object has no attribute 'storage_path' error
        that occurs when the demo project creation tries to access settings.storage_path.
        """
        from app.config import get_settings

        settings = get_settings()
        assert hasattr(settings, 'storage_path'), "Settings must have 'storage_path' attribute"
        assert settings.storage_path is not None, "storage_path cannot be None"
        assert isinstance(settings.storage_path, str), "storage_path must be a string"

    def test_settings_has_required_storage_attributes(self):
        """Ensure all required storage-related attributes exist."""
        from app.config import get_settings

        settings = get_settings()

        required_attrs = ['storage_path', 'upload_dir', 'models_dir']
        for attr in required_attrs:
            assert hasattr(settings, attr), f"Settings must have '{attr}' attribute"
            value = getattr(settings, attr)
            assert value is not None, f"{attr} cannot be None"
            assert isinstance(value, str), f"{attr} must be a string"

    def test_settings_has_database_url(self):
        """Ensure database_url is defined."""
        from app.config import get_settings

        settings = get_settings()
        assert hasattr(settings, 'database_url'), "Settings must have 'database_url' attribute"
        assert settings.database_url is not None, "database_url cannot be None"

    def test_settings_has_security_attributes(self):
        """Ensure security-related attributes exist."""
        from app.config import get_settings

        settings = get_settings()

        required_attrs = ['secret_key', 'algorithm', 'access_token_expire_minutes']
        for attr in required_attrs:
            assert hasattr(settings, attr), f"Settings must have '{attr}' attribute"
            assert getattr(settings, attr) is not None, f"{attr} cannot be None"

    def test_get_settings_returns_same_instance(self):
        """Ensure get_settings returns cached instance (lru_cache)."""
        from app.config import get_settings

        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2, "get_settings should return cached instance"
