#!/usr/bin/env python3
"""
Performance baseline testing for RLCard action processing
Establishes baseline metrics before implementing BB-first action order modifications
"""

import time
import statistics
import sys
import os

# Add RLCard to path
sys.path.insert(0, '/Users/user/Library/Python/3.9/lib/python/site-packages')

try:
    import rlcard
    from rlcard.games.nolimitholdem import game
    from rlcard.games.nolimitholdem.game import NolimitholdemGame
    from rlcard.core import Action
    RLCARD_AVAILABLE = True
except ImportError:
    print("❌ RLCard not available for performance testing")
    RLCARD_AVAILABLE = False


def measure_action_processing_time(game_class, num_games=10, actions_per_game=50):
    """Measure average action processing time"""
    times = []

    for game_num in range(num_games):
        game = game_class()
        state, player_id = game.init_game()

        game_times = []
        actions_taken = 0

        while not game.is_over() and actions_taken < actions_per_game:
            legal_actions = game.get_legal_actions()
            if legal_actions:
                # Use a consistent action for reproducible timing
                action = legal_actions[0]  # Always take first available action

                start_time = time.perf_counter()
                state, next_player = game.step(action)
                end_time = time.perf_counter()

                processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
                game_times.append(processing_time)
                actions_taken += 1
            else:
                break

        if game_times:
            avg_game_time = statistics.mean(game_times)
            times.append(avg_game_time)
            print(".1f"
    if times:
        overall_avg = statistics.mean(times)
        overall_std = statistics.stdev(times) if len(times) > 1 else 0
        return overall_avg, overall_std, times
    else:
        return 0, 0, []


def measure_game_initialization_time(game_class, num_games=100):
    """Measure game initialization time"""
    times = []

    for _ in range(num_games):
        start_time = time.perf_counter()
        game = game_class()
        state, player_id = game.init_game()
        end_time = time.perf_counter()

        init_time = (end_time - start_time) * 1000  # Convert to milliseconds
        times.append(init_time)

    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0

    return avg_time, std_time, times


def measure_memory_usage():
    """Measure memory usage during game play"""
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Play several games
        for _ in range(10):
            game = NolimitholdemGame()
            state, player_id = game.init_game()

            for _ in range(30):
                if game.is_over():
                    break
                legal_actions = game.get_legal_actions()
                if legal_actions:
                    game.step(legal_actions[0])

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = final_memory - initial_memory

        return initial_memory, final_memory, memory_delta

    except ImportError:
        print("⚠️  psutil not available for memory measurement")
        return 0, 0, 0


def run_performance_baseline():
    """Run complete performance baseline testing"""
    if not RLCARD_AVAILABLE:
        print("❌ RLCard not available - cannot run performance baseline")
        return

    print("🔬 Running RLCard Performance Baseline Tests")
    print("=" * 60)

    # Test current RLCard performance
    print("\n📊 Testing Current RLCard Performance:")

    # Action processing time
    print("\n⏱️  Measuring action processing time...")
    avg_action_time, std_action_time, action_times = measure_action_processing_time(
        NolimitholdemGame, num_games=20, actions_per_game=40
    )

    print(".2f"
    # Game initialization time
    print("\n🏁 Measuring game initialization time...")
    avg_init_time, std_init_time, init_times = measure_game_initialization_time(
        NolimitholdemGame, num_games=50
    )

    print(".2f"
    # Memory usage
    print("\n💾 Measuring memory usage...")
    initial_mem, final_mem, mem_delta = measure_memory_usage()

    if mem_delta != 0:
        print(".1f"
    print("\n🎯 Performance Targets for BB-First Implementation:")
    print("- Action processing: < 0.8s (target: < 0.6s)")
    print("- Game initialization: < 10ms")
    print("- Memory usage: < 50MB increase per 10 games")
    print(".2f"    print(".2f"
    # Store baseline results
    baseline_results = {
        'action_processing_avg_ms': avg_action_time,
        'action_processing_std_ms': std_action_time,
        'game_init_avg_ms': avg_init_time,
        'game_init_std_ms': std_init_time,
        'memory_initial_mb': initial_mem,
        'memory_final_mb': final_mem,
        'memory_delta_mb': mem_delta,
        'test_timestamp': time.time(),
        'rlcard_version': rlcard.__version__ if hasattr(rlcard, '__version__') else 'unknown'
    }

    # Save baseline results
    try:
        import json
        with open('performance_baseline.json', 'w') as f:
            json.dump(baseline_results, f, indent=2)
        print("\n💾 Baseline results saved to performance_baseline.json")
    except Exception as e:
        print(f"\n⚠️  Could not save baseline results: {e}")

    # Performance assessment
    print("\n📈 Performance Assessment:")

    if avg_action_time < 100:  # Less than 100ms per action
        print("✅ Action processing performance: EXCELLENT")
    elif avg_action_time < 200:
        print("✅ Action processing performance: GOOD")
    elif avg_action_time < 500:
        print("⚠️  Action processing performance: ACCEPTABLE")
    else:
        print("❌ Action processing performance: CONCERNING")

    if avg_init_time < 10:
        print("✅ Game initialization performance: EXCELLENT")
    elif avg_init_time < 25:
        print("✅ Game initialization performance: GOOD")
    else:
        print("⚠️  Game initialization performance: ACCEPTABLE")

    print("\n🎯 BB-First Implementation Impact Assessment:")
    print("- Expected change: Minimal (only conditional logic addition)")
    print("- Risk level: LOW (no algorithmic changes, only game pointer assignment)")
    print("- Monitoring: Compare post-implementation results against this baseline")

    return baseline_results


if __name__ == '__main__':
    results = run_performance_baseline()

    # Exit with success/failure based on whether we could run tests
    if RLCARD_AVAILABLE:
        print("\n✅ Performance baseline testing completed successfully")
        sys.exit(0)
    else:
        print("\n❌ Performance baseline testing failed - RLCard not available")
        sys.exit(1)
