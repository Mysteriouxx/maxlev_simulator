import sys
from pathlib import Path
from test_indicator_analysis import load_and_prepare_data, analyze_pair_timeframe, compile_simulation_data
from simulator_double import Simulator  # Use double target version

# Add main directory to Python path
main_dir = Path(__file__).parent.parent
sys.path.append(str(main_dir))

def main():
    start_date = '2024-02-01' #inclusive
    end_date = '2024-12-31' #inclusive

    # Load pre-computed results instead of analyzing
    from loading_threshold import load_thresholds
    db_path = "data/thresholds/thresholds_Sim.db"
    all_results = load_thresholds(db_path, start_date, end_date)
    
    if not all_results:
        print("Failed to load threshold results")
        return None

    # Get F3 results from filter.py
    from filter import filter_results
    f3_results = filter_results()
    
    if f3_results is None:
        print("Failed to get F3 results")
        return None
    
    # Compile final results to database with double strategy suffix
    if all_results:
        # Add F3 results to the simulation data
        simulation_data = {
            'analysis_results': all_results,
            'f3_results': f3_results,
            'strategy_mode': 'double'  # Add strategy mode to simulation data
        }
        
        db_path = compile_simulation_data(simulation_data)
        print(f"Final results saved to: {db_path}")
        return db_path
    else:
        print("No results to compile")
        return None

if __name__ == "__main__":
    main() 