"""Tests for the training monitoring module."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime


class TestHTMLMonitorCallback:
    """Test HTML dashboard generation."""

    def test_dashboard_refresh_interval(self):
        """Test dashboard HTML has correct refresh interval."""
        from mission_gym.scripts.monitoring import HTMLMonitorCallback
        
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "test_dashboard.html"
            
            # Create callback with required parameters
            callback = HTMLMonitorCallback(
                html_path=str(html_path),
                run_dir=Path(tmpdir),
            )
            
            # Set required attributes for HTML generation
            callback.start_time = datetime.now()
            callback.num_timesteps = 1000
            
            # Generate HTML (this writes to file)
            callback._generate_html()
            
            # Read the generated HTML
            html = html_path.read_text()
            
            # Check refresh interval is 5 seconds
            assert 'content="5"' in html, "Dashboard should refresh every 5 seconds"
            assert 'Auto-refreshes every 5s' in html, "Footer should indicate 5s refresh"
            
            # Make sure old 30s interval is not present
            assert 'content="30"' not in html, "Old 30s refresh should not be present"
            assert 'every 30s' not in html, "Old 30s text should not be in footer"

    def test_dashboard_contains_required_elements(self):
        """Test dashboard HTML contains essential elements."""
        from mission_gym.scripts.monitoring import HTMLMonitorCallback
        
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "test_dashboard.html"
            
            callback = HTMLMonitorCallback(
                html_path=str(html_path),
                run_dir=Path(tmpdir),
            )
            
            # Set required attributes
            callback.start_time = datetime.now()
            callback.num_timesteps = 5000
            callback.episode_rewards = [10.0, 20.0, 30.0]
            callback.episode_lengths = [100, 150, 200]
            
            callback._generate_html()
            html = html_path.read_text()
            
            # Check essential elements
            assert '<!DOCTYPE html>' in html
            assert '<html lang="en">' in html
            assert 'Mission Gym' in html

    def test_dashboard_meta_refresh_tag_format(self):
        """Test that meta refresh tag is properly formatted."""
        from mission_gym.scripts.monitoring import HTMLMonitorCallback
        
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "test_dashboard.html"
            
            callback = HTMLMonitorCallback(
                html_path=str(html_path),
            )
            callback.start_time = datetime.now()
            callback.num_timesteps = 100
            
            callback._generate_html()
            html = html_path.read_text()
            
            # Check for proper meta refresh format
            assert '<meta http-equiv="refresh" content="5">' in html
