#!/usr/bin/env python3
"""
Update an active training run's dashboard to include the lineage tree.
This is a one-time fix for runs that started before the lineage feature was added.
"""

from pathlib import Path
from mission_gym.scripts.run_utils import build_lineage_tree_html, get_runs_dir
import json
import sys

def update_dashboard_with_lineage(run_dir: Path):
    """Add lineage tree panel to an existing dashboard HTML."""
    
    dashboard_file = run_dir / "dashboard.html"
    if not dashboard_file.exists():
        print(f"❌ Dashboard not found: {dashboard_file}")
        return False
    
    # Read current HTML
    html_content = dashboard_file.read_text()
    
    # Check if lineage tree already exists
    if "Policy Lineage Tree" in html_content:
        print(f"✓ Dashboard already has lineage tree panel")
        return True
    
    print(f"📝 Adding lineage tree to dashboard...")
    
    # Collect all runs for lineage tree
    runs_dir = get_runs_dir()
    runs_data = []
    
    for rd in runs_dir.iterdir():
        if not rd.is_dir():
            continue
        
        lineage_file = rd / "lineage.json"
        metadata_file = rd / "run_metadata.json"
        
        if not lineage_file.exists() and not metadata_file.exists():
            continue
        
        run_info = {
            "name": rd.name,
            "parent": None,
            "created": None,
            "timesteps": 0,
            "lineage": {},
        }
        
        if lineage_file.exists():
            try:
                with open(lineage_file) as f:
                    lineage = json.load(f)
                    run_info["parent"] = lineage.get("parent_run_name")
                    run_info["created"] = lineage.get("created_at")
                    run_info["lineage"] = lineage
            except Exception:
                pass
        
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    run_info["timesteps"] = metadata.get("args", {}).get("timesteps", 0)
                    if not run_info["created"]:
                        run_info["created"] = metadata.get("created_at")
            except Exception:
                pass
        
        runs_data.append(run_info)
    
    # Generate lineage tree HTML
    lineage_tree_html = build_lineage_tree_html(runs_data, run_dir.name)
    
    lineage_panel = f'''
        <!-- Policy Lineage Tree -->
        <div class="panel" style="margin-bottom: 2rem;">
            <div class="panel-header">
                <div class="panel-title"><span class="icon">🌳</span> Policy Lineage Tree</div>
            </div>
            <div class="panel-body">
                <div style="font-size: 0.85rem; line-height: 1.8; font-family: 'JetBrains Mono', monospace;">
                    {lineage_tree_html}
                </div>
            </div>
        </div>
        
        <!-- Configuration -->'''
    
    # Insert lineage panel before Configuration section
    updated_html = html_content.replace(
        '<!-- Configuration -->',
        lineage_panel
    )
    
    # Add CSS if not present
    if '.lineage-tree' not in updated_html:
        css_insert = '''
        /* Lineage Tree */
        .lineage-tree {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            line-height: 1.8;
        }
        
        .tree-node {
            margin: 0.25rem 0;
        }
        
        .tree-node.current-run {
            background: rgba(0, 212, 170, 0.1);
            border-left: 3px solid var(--accent-teal);
            padding-left: 0.5rem;
            margin-left: -0.5rem;
        }
        
        .tree-line {
            color: var(--text-secondary);
        }
        
        .run-name {
            color: var(--text-primary);
            font-weight: 500;
        }
        
        .run-meta {
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-left: 0.5rem;
        }
        
        .tree-notes {
            color: var(--accent-cyan);
            font-size: 0.8rem;
            margin-top: 0.25rem;
            font-style: italic;
        }
        
        /* Reward Components */'''
        
        updated_html = updated_html.replace(
            '/* Reward Components */',
            css_insert
        )
    
    # Write updated HTML
    dashboard_file.write_text(updated_html)
    print(f"✓ Lineage tree added to dashboard")
    
    # Also fix the Quick Commands to use actual latest checkpoint
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        checkpoints = sorted(checkpoints_dir.glob("ppo_mission_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if checkpoints:
            latest_checkpoint = checkpoints[0]
            
            # Fix the Evaluate Model command
            old_eval_cmd = f"python -m mission_gym.scripts.evaluate --model {run_dir}/final_model"
            new_eval_cmd = f"python -m mission_gym.scripts.evaluate --model {latest_checkpoint.with_suffix('')}"
            
            if old_eval_cmd in updated_html:
                updated_html = updated_html.replace(old_eval_cmd, new_eval_cmd)
                dashboard_file.write_text(updated_html)
                print(f"✓ Fixed Evaluate command to use latest checkpoint: {latest_checkpoint.name}")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        # Default to most recently updated run
        runs_dir = get_runs_dir()
        runs = []
        for rd in runs_dir.iterdir():
            if rd.is_dir() and (rd / "dashboard.html").exists():
                runs.append((rd, (rd / "dashboard.html").stat().st_mtime))
        
        if not runs:
            print("❌ No training runs found")
            sys.exit(1)
        
        runs.sort(key=lambda x: x[1], reverse=True)
        run_dir = runs[0][0]
    
    print(f"🎯 Updating dashboard: {run_dir.name}")
    
    if update_dashboard_with_lineage(run_dir):
        print(f"\n✨ Success! Refresh your browser to see the changes.")
        print(f"   Dashboard: file://{run_dir}/dashboard.html")
    else:
        print(f"\n❌ Failed to update dashboard")
        sys.exit(1)
