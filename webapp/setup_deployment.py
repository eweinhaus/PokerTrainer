#!/usr/bin/env python3
"""
Deployment setup script for Render.
Copies modified RLCard game.py to override the standard RLCard installation.
"""

import os
import shutil
import sys
from pathlib import Path

def setup_rlcard_override():
    """Copy modified game.py to RLCard installation directory."""
    try:
        # Get the path to the modified game.py
        current_dir = Path(__file__).parent
        modified_game_path = current_dir.parent / "rlcard_modified" / "game.py"

        if not modified_game_path.exists():
            print(f"ERROR: Modified game.py not found at {modified_game_path}")
            return False

        # Find RLCard installation directory
        # Try to import rlcard and find its location
        try:
            import rlcard
            rlcard_path = Path(rlcard.__file__).parent
            games_path = rlcard_path / "games" / "nolimitholdem"
            target_path = games_path / "game.py"

            print(f"Found RLCard at: {rlcard_path}")
            print(f"Target path: {target_path}")

            if not games_path.exists():
                print(f"ERROR: RLCard nolimitholdem games directory not found: {games_path}")
                return False

            # Backup original if it exists
            if target_path.exists():
                backup_path = target_path.with_suffix('.py.backup')
                shutil.copy2(target_path, backup_path)
                print(f"Backed up original game.py to {backup_path}")

            # Copy modified game.py
            shutil.copy2(modified_game_path, target_path)
            print(f"Successfully copied modified game.py to {target_path}")

            return True

        except ImportError as e:
            print(f"ERROR: Could not import rlcard: {e}")
            return False

    except Exception as e:
        print(f"ERROR: Failed to setup RLCard override: {e}")
        return False

if __name__ == "__main__":
    print("Setting up RLCard override for deployment...")
    success = setup_rlcard_override()
    if success:
        print("RLCard override setup completed successfully!")
        sys.exit(0)
    else:
        print("RLCard override setup failed!")
        sys.exit(1)
