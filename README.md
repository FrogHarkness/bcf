# Peer Reviewer Assignment Tool

This tool helps conference/journal organizers assign reviewers to papers based on expertise and preferences.

## Features

- Assign reviewers to papers based on preference scores
- Customize each reviewer's maximum capacity individually
- Find backup reviewers for each paper
- Generate Excel reports with detailed assignment information
- User-friendly GUI interface

## Requirements

- Python 3.7 or higher
- Required packages: pandas, numpy, gurobipy, tkinter, cx_Freeze (for building executable)

## Installation

1. Install required packages:
   ```
   pip install pandas numpy gurobipy cx_Freeze
   ```

2. Run the script directly:
   ```
   python prer_.py
   ```

3. Or build an executable:
   ```
   python build_exe.py build
   ```

## Usage

1. **Prepare your input Excel file**: 
   - First column must contain reviewer names
   - Column headers should be paper names/IDs
   - Cells should contain preference ratings: "Moderate", "Considerable", etc.
   - Include a "Reviewers" column (will be dropped internally)

2. **Launch the application**:
   - Run the Python script or the executable

3. **Use the GUI**:
   - Click "Browse" to select your Excel file
   - Click "Load Reviewers" to load reviewer names
   - Double-click on a reviewer's capacity to adjust it
   - Set "Reviewers per Paper" value (default is 2)
   - Click "Run Assignment" to generate assignments
   - Results will be saved to "final_result.xlsx"

## Output Files

The output Excel file contains:
- Reviewer Assignments: Assigned reviewers for each paper
- Raw Assignments: Preference scores for assignments
- Assignment Summary: Number of papers per reviewer
- Paper Scores: Overall score for each paper based on reviewer preferences 