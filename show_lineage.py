#!/usr/bin/env python3
"""
Display policy lineage tree for training runs.

Usage:
    python show_lineage.py                    # Show all lineage trees
    python show_lineage.py RUN_NAME           # Show specific run's lineage
    python show_lineage.py --active           # Show only active lineage tree
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime

from mission_gym.scripts.run_utils import get_runs_dir


def load_run_info(run_dir: Path) -> Optional[Dict]:
    """Load lineage and metadata for a run."""
    lineage_file = run_dir / "lineage.json"
    metadata_file = run_dir / "run_metadata.json"
    
    info = {
        "name": run_dir.name,
        "parent": None,
        "created": None,
        "timesteps": 0,
        "notes": None,
        "branch_name": None,
    }
    
    if lineage_file.exists():
        try:
            with open(lineage_file) as f:
                lineage = json.load(f)
                info["parent"] = lineage.get("parent_run_name")
                info["created"] = lineage.get("created_at")
                info["notes"] = lineage.get("notes")
                info["branch_name"] = lineage.get("branch_name")
        except Exception:
            pass
    
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
                info["timesteps"] = metadata.get("args", {}).get("timesteps", 0)
                if not info["created"]:
                    info["created"] = metadata.get("created_at")
        except Exception:
            pass
    
    return info


def build_tree_structure(runs: List[Dict]) -> Dict[str, List[str]]:
    """Build parent -> children mapping."""
    children = {}
    for run in runs:
        parent = run.get("parent")
        if parent:
            if parent not in children:
                children[parent] = []
            children[parent].append(run["name"])
    return children


def find_roots(runs: List[Dict]) -> List[str]:
    """Find root runs (no parent or parent doesn't exist)."""
    run_names = {r["name"] for r in runs}
    roots = []
    for run in runs:
        parent = run.get("parent")
        if not parent or parent not in run_names:
            roots.append(run["name"])
    return roots


def print_tree(
    run_name: str,
    runs_by_name: Dict[str, Dict],
    children: Dict[str, List[str]],
    prefix: str = "",
    is_last: bool = True,
):
    """Recursively print tree structure."""
    run = runs_by_name.get(run_name)
    if not run:
        return
    
    # Format run info
    connector = "└── " if is_last else "├── "
    name = run["name"]
    
    # Add metadata
    created = run.get("created", "")
    if created:
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            created_str = dt.strftime("%Y-%m-%d %H:%M")
        except:
            created_str = created[:16] if len(created) > 16 else created
    else:
        created_str = "unknown"
    
    timesteps = run.get("timesteps", 0)
    timesteps_str = f"{timesteps/1_000_000:.1f}M" if timesteps >= 1_000_000 else f"{timesteps/1000:.0f}K"
    
    # Build line
    line = f"{prefix}{connector}{name}"
    line += f" [{created_str}, {timesteps_str} steps]"
    
    # Add notes if present
    notes = run.get("notes")
    if notes:
        line += f"\n{prefix}{'    ' if is_last else '│   '}   ➜ {notes}"
    
    print(line)
    
    # Print children
    child_names = children.get(run_name, [])
    for i, child_name in enumerate(child_names):
        is_last_child = (i == len(child_names) - 1)
        child_prefix = prefix + ("    " if is_last else "│   ")
        print_tree(child_name, runs_by_name, children, child_prefix, is_last_child)


def main():
    parser = argparse.ArgumentParser(description="Display policy lineage tree")
    parser.add_argument("run_name", nargs="?", help="Show lineage for specific run")
    parser.add_argument("--active", action="store_true", help="Show only active lineage tree")
    args = parser.parse_args()
    
    runs_dir = get_runs_dir()
    
    # Load all runs
    all_runs = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        dashboard_file = run_dir / "dashboard.html"
        if not dashboard_file.exists():
            continue
        
        info = load_run_info(run_dir)
        if info:
            # Add modification time for sorting
            info["mtime"] = dashboard_file.stat().st_mtime
            all_runs.append(info)
    
    if not all_runs:
        print("No training runs found.")
        return
    
    # Sort by modification time (most recent first)
    all_runs.sort(key=lambda x: x["mtime"], reverse=True)
    
    # Build structures
    runs_by_name = {r["name"]: r for r in all_runs}
    children = build_tree_structure(all_runs)
    roots = find_roots(all_runs)
    
    # Determine what to show
    if args.run_name:
        # Show specific run's lineage (ancestors + descendants)
        if args.run_name not in runs_by_name:
            print(f"Run '{args.run_name}' not found.")
            return
        
        # Find root ancestor
        current = args.run_name
        visited = set()
        while current and current not in visited:
            visited.add(current)
            parent = runs_by_name.get(current, {}).get("parent")
            if parent and parent in runs_by_name:
                current = parent
            else:
                break
        
        root = current
        print(f"\n🌳 Policy Lineage for: {args.run_name}\n")
        print_tree(root, runs_by_name, children)
        
    elif args.active:
        # Show only active lineage tree (most recent)
        most_recent = all_runs[0]["name"]
        
        # Find root
        current = most_recent
        visited = set()
        while current and current not in visited:
            visited.add(current)
            parent = runs_by_name.get(current, {}).get("parent")
            if parent and parent in runs_by_name:
                current = parent
            else:
                break
        
        root = current
        print(f"\n🌳 Active Policy Lineage (Most Recent: {most_recent})\n")
        print_tree(root, runs_by_name, children)
        
    else:
        # Show all lineage trees
        print("\n🌳 All Policy Lineage Trees\n")
        for root in roots:
            print_tree(root, runs_by_name, children)
            print()  # Blank line between trees
    
    # Summary
    total_runs = len(all_runs)
    num_trees = len(roots)
    print(f"\n📊 Summary: {total_runs} runs in {num_trees} lineage tree(s)")


if __name__ == "__main__":
    main()
