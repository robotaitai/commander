"""Tests for the training monitoring module."""

import pytest
import re
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

    def test_dashboard_shows_run_name(self):
        """Test that run name is displayed in dashboard."""
        from mission_gym.scripts.monitoring import HTMLMonitorCallback
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a run directory with a properly formatted name
            run_name = "swift-falcon-20260121-143052"
            run_dir = Path(tmpdir) / run_name
            run_dir.mkdir(parents=True)
            html_path = run_dir / "dashboard.html"
            
            callback = HTMLMonitorCallback(
                html_path=str(html_path),
                run_dir=run_dir,
            )
            callback.start_time = datetime.now()
            callback.num_timesteps = 1000
            
            callback._generate_html()
            html = html_path.read_text()
            
            # Run name should appear in header badge
            assert 'run-badge' in html, "Dashboard should have run-badge element"
            assert run_name in html, "Run name should appear in dashboard"
            # Run name should appear in title
            assert f'<title>{run_name}' in html, "Run name should be in page title"

    def test_dashboard_action_log(self):
        """Test that action log is displayed in dashboard."""
        from mission_gym.scripts.monitoring import HTMLMonitorCallback
        
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "test_dashboard.html"
            
            callback = HTMLMonitorCallback(
                html_path=str(html_path),
                run_dir=Path(tmpdir),
            )
            callback.start_time = datetime.now()
            callback.num_timesteps = 1000
            
            # Add some action log entries
            for step in range(50):
                callback.action_log.append({
                    'timestep': 950 + step,
                    'actions': [0, 1, 2, 3],  # 4 units
                    'rewards': [0.1, -0.2, 0.3, -0.1],
                    'dones': [False, False, False, False],
                })
            
            callback._generate_html()
            html = html_path.read_text()
            
            # Action log should be present
            assert 'Action Log' in html, "Dashboard should have Action Log section"
            assert 'action-badge' in html, "Dashboard should have action badges"
            assert '50 recorded' in html, "Should show number of recorded entries"

    def test_action_log_empty_state(self):
        """Test action log shows empty state when no actions recorded."""
        from mission_gym.scripts.monitoring import HTMLMonitorCallback
        
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "test_dashboard.html"
            
            callback = HTMLMonitorCallback(
                html_path=str(html_path),
                run_dir=Path(tmpdir),
            )
            callback.start_time = datetime.now()
            callback.num_timesteps = 100
            
            # No action log entries added
            callback._generate_html()
            html = html_path.read_text()
            
            # Should show empty state message
            assert 'Action Log' in html, "Dashboard should have Action Log section"
            assert 'No actions recorded yet' in html, "Should show empty state message"


class TestRunNameGeneration:
    """Test run name generation and format."""

    def test_generate_run_name_format(self):
        """Test auto-generated run name follows word-word-timestamp format."""
        from mission_gym.scripts.run_utils import generate_run_name
        
        name = generate_run_name()
        
        # Should match pattern: word-word-YYYYMMDD-HHMMSS
        pattern = r'^[a-z]+-[a-z]+-\d{8}-\d{6}$'
        assert re.match(pattern, name), f"Run name '{name}' should match pattern word-word-timestamp"

    def test_create_run_dir_with_custom_name_appends_timestamp(self):
        """Test custom run names get timestamp appended."""
        from mission_gym.scripts.run_utils import create_run_dir
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override runs directory
            import mission_gym.scripts.run_utils as run_utils
            original_get_runs_dir = run_utils.get_runs_dir
            run_utils.get_runs_dir = lambda: Path(tmpdir)
            
            try:
                run_dir = create_run_dir("my-experiment")
                run_name = run_dir.name
                
                # Should start with custom prefix and end with timestamp
                assert run_name.startswith("my-experiment-"), f"Run name should start with custom prefix"
                # Should have timestamp at end (YYYYMMDD-HHMMSS)
                pattern = r'^my-experiment-\d{8}-\d{6}$'
                assert re.match(pattern, run_name), f"Run name '{run_name}' should have timestamp appended"
            finally:
                run_utils.get_runs_dir = original_get_runs_dir

    def test_create_run_dir_auto_generates_name(self):
        """Test auto-generated run name when no name provided."""
        from mission_gym.scripts.run_utils import create_run_dir
        
        with tempfile.TemporaryDirectory() as tmpdir:
            import mission_gym.scripts.run_utils as run_utils
            original_get_runs_dir = run_utils.get_runs_dir
            run_utils.get_runs_dir = lambda: Path(tmpdir)
            
            try:
                run_dir = create_run_dir(None)
                run_name = run_dir.name
                
                # Should match word-word-timestamp pattern
                pattern = r'^[a-z]+-[a-z]+-\d{8}-\d{6}$'
                assert re.match(pattern, run_name), f"Auto-generated name '{run_name}' should match pattern"
            finally:
                run_utils.get_runs_dir = original_get_runs_dir
