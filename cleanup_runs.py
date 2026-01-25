#!/usr/bin/env python3
"""
Cleanup script for training runs.

Usage:
    # Keep only runs in the best lineage tree (most recent active run)
    python cleanup_runs.py --keep-active-lineage
    
    # Keep specific run and its lineage
    python cleanup_runs.py --keep-lineage branch-20260122-193222
    
    # Delete all failed runs (0 checkpoints)
    python cleanup_runs.py --delete-failed
    
    # Interactive mode - prompts for each run
    python cleanup_runs.py --interactive
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Set, List, Dict

from mission_gym.scripts.run_utils import get_runs_dir


def get_run_info(run_dir: Path) -> Dict:
    """Get information about a run."""
    metadata_file = run_dir / "run_metadata.json"
    lineage_file = run_dir / "lineage.json"
    checkpoints_dir = run_dir / "checkpoints"
    
    metadata = {}
    lineage = {}
    parent_run = None
    
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
        except Exception:
            pass
    
    if lineage_file.exists():
        try:
            with open(lineage_file) as f:
                lineage = json.load(f)
                parent_run = lineage.get("parent_run_name")
        except Exception:
            pass
    
    # Count checkpoints
    num_checkpoints = 0
    if checkpoints_dir.exists():
        num_checkpoints = len(list(checkpoints_dir.glob("*.zip")))
    
    # Get total size
    size_bytes = sum(f.stat().st_size for f in run_dir.rglob('*') if f.is_file())
    size_mb = size_bytes / (1024 * 1024)
    
    return {
        "name": run_dir.name,
        "path": run_dir,
        "parent": parent_run,
        "checkpoints": num_checkpoints,
        "size_mb": size_mb,
        "timesteps": metadata.get("args", {}).get("timesteps", 0),
        "created": metadata.get("created_at", ""),
    }


def get_lineage_tree(root_run: str, all_runs: List[Dict]) -> Set[str]:
    """Get all runs in a lineage tree (root + all descendants)."""
    tree = {root_run}
    
    # Find all descendants
    changed = True
    while changed:
        changed = False
        for run in all_runs:
            parent = run.get("parent")
            if parent and parent in tree and run["name"] not in tree:
                tree.add(run["name"])
                changed = True
    
    # Find all ancestors
    current = root_run
    visited = set()
    while current and current not in visited:
        visited.add(current)
        run_data = next((r for r in all_runs if r["name"] == current), None)
        if run_data and run_data.get("parent"):
            parent = run_data["parent"]
            tree.add(parent)
            current = parent
        else:
            break
    
    return tree


def main():
    parser = argparse.ArgumentParser(description="Cleanup training runs")
    parser.add_argument(
        "--keep-active-lineage",
        action="store_true",
        help="Keep only the most recent lineage tree (based on latest dashboard update)",
    )
    parser.add_argument(
        "--keep-lineage",
        type=str,
        help="Keep specific run and its entire lineage tree",
    )
    parser.add_argument(
        "--delete-failed",
        action="store_true",
        help="Delete runs with 0 checkpoints (failed immediately)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for each run before deleting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    
    args = parser.parse_args()
    
    runs_dir = get_runs_dir()
    
    # Collect all runs
    all_runs = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        dashboard_file = run_dir / "dashboard.html"
        if not dashboard_file.exists():
            continue
        
        info = get_run_info(run_dir)
        info["mtime"] = dashboard_file.stat().st_mtime
        all_runs.append(info)
    
    if not all_runs:
        print("No runs found.")
        return
    
    # Sort by modification time (most recent first)
    all_runs.sort(key=lambda x: x["mtime"], reverse=True)
    
    # Determine which runs to keep
    runs_to_keep = set()
    
    if args.keep_active_lineage:
        # Keep the lineage tree of the most recent run
        most_recent = all_runs[0]["name"]
        runs_to_keep = get_lineage_tree(most_recent, all_runs)
        print(f"\n🔒 Keeping active lineage tree (root: {most_recent})")
        print(f"   Runs in tree: {', '.join(sorted(runs_to_keep))}\n")
    
    elif args.keep_lineage:
        # Keep specific lineage tree
        runs_to_keep = get_lineage_tree(args.keep_lineage, all_runs)
        print(f"\n🔒 Keeping lineage tree for: {args.keep_lineage}")
        print(f"   Runs in tree: {', '.join(sorted(runs_to_keep))}\n")
    
    elif args.delete_failed:
        # Keep all runs with checkpoints
        runs_to_keep = {r["name"] for r in all_runs if r["checkpoints"] > 0}
        print(f"\n🔒 Keeping runs with checkpoints ({len(runs_to_keep)} runs)\n")
    
    # Determine what to delete
    runs_to_delete = []
    for run in all_runs:
        if run["name"] not in runs_to_keep:
            runs_to_delete.append(run)
    
    if not runs_to_delete:
        print("✅ No runs to delete.")
        return
    
    # Show summary
    total_size_mb = sum(r["size_mb"] for r in runs_to_delete)
    print(f"📊 Summary:")
    print(f"   Total runs: {len(all_runs)}")
    print(f"   Runs to keep: {len(runs_to_keep)}")
    print(f"   Runs to delete: {len(runs_to_delete)}")
    print(f"   Space to free: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)\n")
    
    # Show what will be deleted
    print("🗑️  Runs to delete:\n")
    for run in runs_to_delete:
        status = "FAILED" if run["checkpoints"] == 0 else f"{run['checkpoints']} ckpts"
        print(f"   • {run['name']}")
        print(f"     Size: {run['size_mb']:.1f} MB | {status} | {run['timesteps']:,} steps")
    
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN - No files were deleted.")
        return
    
    # Confirm deletion
    if not args.interactive:
        response = input("⚠️  Delete these runs? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Cancelled.")
            return
    
    # Delete runs
    deleted_count = 0
    for run in runs_to_delete:
        if args.interactive:
            response = input(f"Delete {run['name']} ({run['size_mb']:.1f} MB)? [y/N]: ")
            if response.lower() != 'y':
                print(f"   ⏭️  Skipped")
                continue
        
        try:
            shutil.rmtree(run["path"])
            print(f"   ✓ Deleted {run['name']}")
            deleted_count += 1
        except Exception as e:
            print(f"   ✗ Error deleting {run['name']}: {e}")
    
    print(f"\n✅ Deleted {deleted_count} run(s), freed {total_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
