import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
import os

# Default mapping of preference strings to scores
DEFAULT_PREFERENCE_SCORES = {
    "Conflict of Interest": -100,
    "None": -100,
    "Moderate": 1,
    "Moderate*": 1,
    "Moderate**": 1,
    "Considerable": 5,
    "Considerable*": 5,
    "Considerable**": 5
}

def run_assignment(input_file, reviewer_capacities=None, demand=2, preference_scores=None):
    df = pd.read_excel(input_file)
    
    df = df.drop(columns=['Reviewers'])
    
    M = df.shape[1]  # Number of papers
    N = df.shape[0]  # Number of reviewers
    
    simmatrix = np.zeros((M, N), dtype=float)
    
    # Use provided preferences or defaults
    if preference_scores is None:
        PREFERENCE_SCORES = DEFAULT_PREFERENCE_SCORES.copy()
    else:
        # copy to avoid mutating caller data
        PREFERENCE_SCORES = preference_scores.copy()
    
    for paper_idx in range(M):
        for rev_idx in range(N):
            cell_value = df.iat[rev_idx, paper_idx]
            if pd.isna(cell_value) or str(cell_value).strip() == "":
                # If the cell is empty, treat as "None"
                score_str = "None"
            else:
                score_str = str(cell_value).strip()
    
            # Lookup the score, default to -1 if not in map
            score = PREFERENCE_SCORES.get(score_str, -100)
            simmatrix[paper_idx, rev_idx] = score
    
    np.set_printoptions(threshold=np.inf)
    simmatrix = np.transpose(simmatrix)  # always transpose the matrix
    
    # If no custom capacities provided, use default for all reviewers
    if reviewer_capacities is None:
        default_capacity = 8
        reviewer_capacities = [default_capacity] * N
    
    # Make sure we have the right number of capacities
    if len(reviewer_capacities) < N:
        reviewer_capacities.extend([8] * (N - len(reviewer_capacities)))
    elif len(reviewer_capacities) > N:
        reviewer_capacities = reviewer_capacities[:N]
        
    # Run the assignment algorithm
    res = auto_assigner(
        simmatrix=simmatrix,
        demand=demand,
        ability=reviewer_capacities,  # Use individual capacities
        function=lambda x: x,
        iter_limit=9999000,
        time_limit=99900
    ).fair_assignment()
    
    # Calculate the number of papers assigned to each reviewer
    reviewer_counts = {}
    for paper, reviewers in res.items():
        for reviewer in reviewers:
            if reviewer in reviewer_counts:
                reviewer_counts[reviewer] += 1
            else:
                reviewer_counts[reviewer] = 1
    
    # Find 4 backup reviewers for each paper
    backup_reviewers = {}
    for paper_idx in range(len(df.columns)):
        if paper_idx in res:
            # Get the current assigned reviewers
            assigned_reviewers = res[paper_idx]
            
            # Get scores for all reviewers for this paper
            paper_scores = []
            for rev_idx in range(N):
                if rev_idx not in assigned_reviewers and simmatrix[rev_idx, paper_idx] >= 0:
                    # Calculate score and include reviewer's current workload
                    score = simmatrix[rev_idx, paper_idx]
                    paper_scores.append((rev_idx, score, reviewer_counts.get(rev_idx, 0)))
                    
            # Sort by score (descending)
            paper_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 4
            backup_reviewers[paper_idx] = paper_scores[:4]
    
    # Create assignment and names DataFrames with ordered first two by preference
    ndf = pd.DataFrame(columns=df.columns)
    nameevodf = pd.DataFrame(columns=df.columns)
    withnameoriginaldf = pd.read_excel(input_file)
    for col_idx, reviewers in res.items():
        if col_idx < len(df.columns):
            col_name = df.columns[col_idx]
            r1, r2 = reviewers
            # Order by numeric preference score
            if simmatrix[r2, col_idx] > simmatrix[r1, col_idx]:
                r1, r2 = r2, r1
            ndf.at[0, col_name] = df.iloc[r1, col_idx]
            ndf.at[1, col_name] = df.iloc[r2, col_idx]
            nameevodf.at[0, col_name] = withnameoriginaldf.iloc[r1, 0]
            nameevodf.at[1, col_name] = withnameoriginaldf.iloc[r2, 0]
    
    # Add backup reviewers
    for paper_idx, backups in backup_reviewers.items():
        if paper_idx < len(df.columns):
            col_name = df.columns[paper_idx]
            for i, (rev_idx, score, count) in enumerate(backups):
                nameevodf.at[i+2, col_name] = f"{withnameoriginaldf.iloc[rev_idx, 0]} ({count})"
                ndf.at[i+2, col_name] = f"{df.iloc[rev_idx, paper_idx]} ({count})"
    
    # Insert blank separator row and label rows
    sep_name = pd.DataFrame([[''] * len(df.columns)], columns=df.columns)
    sep_num  = pd.DataFrame([[''] * len(df.columns)], columns=df.columns)
    nameevodf = pd.concat([nameevodf.iloc[:2], sep_name, nameevodf.iloc[2:]], ignore_index=True)
    ndf        = pd.concat([ndf.iloc[:2],      sep_num,  ndf.iloc[2:]],      ignore_index=True)
    labels = ['Assigner Rank 1', 'Assigner Rank 2', '', 'Backup 1', 'Backup 2', 'Backup 3', 'Backup 4']
    nameevodf.index = labels
    ndf.index        = labels
    
    # Create a summary table of reviewer assignments
    summary_df = pd.DataFrame(columns=["Reviewer", "Number of Assignments", "Capacity"])
    row_idx = 0
    for reviewer, count in reviewer_counts.items():
        summary_df.at[row_idx, "Reviewer"] = withnameoriginaldf.iloc[reviewer, 0]
        summary_df.at[row_idx, "Number of Assignments"] = count
        summary_df.at[row_idx, "Capacity"] = reviewer_capacities[reviewer]
        row_idx += 1
    
    # Sort by number of assignments (descending)
    summary_df = summary_df.sort_values(by="Number of Assignments", ascending=False).reset_index(drop=True)
    
    # Calculate overall scores for each paper
    paper_scores_df = pd.DataFrame(columns=["Paper", "Overall Score", "Assigned Reviewers"])
    for paper_idx, reviewers in res.items():
        if paper_idx < len(df.columns):
            paper_name = df.columns[paper_idx]
            overall_score = sum(simmatrix[reviewer, paper_idx] for reviewer in reviewers)
            assigned_reviewers = ", ".join([withnameoriginaldf.iloc[reviewer, 0] for reviewer in reviewers])
            
            paper_scores_df = pd.concat([paper_scores_df, pd.DataFrame({
                "Paper": [paper_name],
                "Overall Score": [overall_score],
                "Assigned Reviewers": [assigned_reviewers]
            })], ignore_index=True)
    
    # Sort papers by overall score (descending)
    paper_scores_df = paper_scores_df.sort_values(by="Overall Score").reset_index(drop=True)
    
    # Print overall scores
    print("\nOverall Paper Scores (Sum of Assigned Reviewers' Preference Scores):")
    print(paper_scores_df)
    print(f"\nMinimum Paper Score: {paper_scores_df['Overall Score'].min()}")
    print(f"Average Paper Score: {paper_scores_df['Overall Score'].mean():.2f}")
    
    # Derive output filename based on input file
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_filename = f"{base_name}_pr_result.xlsx"
    # Create a writer to save multiple sheets
    with pd.ExcelWriter(output_filename) as writer:
        nameevodf.to_excel(writer, sheet_name='Reviewer Assignments', index=True)
        ndf.to_excel(writer, sheet_name='Raw Assignments', index=True)
        summary_df.to_excel(writer, sheet_name='Assignment Summary')
        paper_scores_df.to_excel(writer, sheet_name='Paper Scores')
    
    return res, summary_df, paper_scores_df, output_filename

# Keep the auto_assigner class as is
from gurobipy import Model, GRB
import numpy as np
import time
from itertools import product

class auto_assigner:
    _EPS = 1e-3
    
    def __init__(self, simmatrix, demand=2, ability=100, function=lambda x: x,
                 iter_limit=np.inf, time_limit=np.inf):
        self.simmatrix = simmatrix
        self.numrev = simmatrix.shape[0]
        self.numpapers = simmatrix.shape[1]
        
        # If ability is just an integer, replicate for all reviewers
        if isinstance(ability, int):
            self.ability = ability * np.ones(self.numrev, dtype=int)
        else:
            self.ability = np.array(ability, dtype=int)
        
        self.demand = demand
        self.function = function
        if iter_limit < 1:
            raise ValueError('Maximum number of iterations must be at least 1')
        self.iter_limit = iter_limit
        self.time_limit = time_limit
        
    def _initialize_model(self):
        self._problem = Model()
        self._problem.setParam('OutputFlag', False)

        # Source to reviewers capacity
        self._source_vars = self._problem.addVars(
            self.numrev, vtype=GRB.CONTINUOUS, 
            lb=0.0, ub=self.ability, name='reviewers'
        )

        # Papers to sink capacity
        self._sink_vars = self._problem.addVars(
            self.numpapers, vtype=GRB.CONTINUOUS,
            lb=0.0, ub=self.demand, name='papers'
        )

        # *** Use BINARY assignment variables so each (reviewer,paper) is either 0 or 1
        self._mix_vars = self._problem.addVars(
            self.numrev, self.numpapers, 
            vtype=GRB.BINARY, 
            lb=0.0, ub=1.0,
            name='assignment'
        )
        self._problem.update()

        # Flow balance constraints
        self._balance_reviewers = self._problem.addConstrs(
            (self._source_vars[i] == self._mix_vars.sum(i, '*') 
             for i in range(self.numrev))
        )
        self._balance_papers = self._problem.addConstrs(
            (self._sink_vars[i] == self._mix_vars.sum('*', i)
             for i in range(self.numpapers))
        )
        self._problem.update()

    def _ranking_of_pairs(self, simmatrix):
        # Only consider pairs where score >= 0
        pairs = [(r, p) for r in range(self.numrev)
                 for p in range(self.numpapers)
                 if simmatrix[r, p] >= 0]
        return sorted(pairs, key=lambda x: simmatrix[x[0], x[1]], reverse=True)

    def _subroutine(self, simmatrix, kappa, abilities, not_assigned, lower_bound, *args):
 

        # 1. First objective: maximize total flow (just to ensure we can get kappa * #papers).
        self._problem.setObjective(
            sum(self._source_vars[i] for i in range(self.numrev)),
            GRB.MAXIMIZE
        )

        # 2. Papers not yet assigned get an upper bound = kappa (we want exactly kappa, but LB=0 for now)
        for paper in not_assigned:
            self._sink_vars[paper].ub = kappa
            self._sink_vars[paper].lb = 0

        # 3. Update each reviewer's capacity
        for reviewer in range(self.numrev):
            self._source_vars[reviewer].ub = abilities[reviewer]

        # 4. Get the sorted (reviewer, paper) pairs by descending sim score (only ≥ 0)
        sorted_pairs = self._ranking_of_pairs(simmatrix)

        # If an upper bound was passed in via *args, use it; otherwise default
        if args != ():
            upper_bound = args[0]
        else:
            upper_bound = len(sorted_pairs)

        current_solution = 0
        maxflow = -1
        one_iteration_done = False

        while lower_bound < upper_bound or not one_iteration_done:
            one_iteration_done = True
            prev_solution = current_solution
            current_solution = lower_bound + (upper_bound - lower_bound) // 2

            if current_solution == prev_solution:
                if maxflow < len(not_assigned) * kappa and current_solution == lower_bound:
                    current_solution += 1
                    lower_bound += 1
                else:
                    raise ValueError('An error occured1')

            if current_solution > prev_solution:
                for cur_pair in sorted_pairs[prev_solution : current_solution]:
                    self._mix_vars[cur_pair[0], cur_pair[1]].ub = 1
            else:
                for cur_pair in sorted_pairs[current_solution : prev_solution]:
                    self._mix_vars[cur_pair[0], cur_pair[1]].ub = 0

            self._problem.optimize()
            maxflow = self._problem.objVal

            if maxflow == len(not_assigned) * kappa:
                upper_bound = current_solution
            elif maxflow < len(not_assigned) * kappa:
                lower_bound = current_solution
            else:
                raise ValueError('An error occured2')

        if maxflow != len(not_assigned) * kappa or lower_bound != current_solution:
            print(f"Debug Info: maxflow={maxflow}, required_flow={len(not_assigned) * kappa}, lower_bound={lower_bound}, current_solution={current_solution}")
            raise ValueError('An error occured3')

        for paper in not_assigned:
            self._sink_vars[paper].lb = kappa

        self._problem.setObjective(sum([sum([simmatrix[reviewer, paper] * self._mix_vars[reviewer, paper]
                                            for paper in not_assigned])
                                        for reviewer in range(self.numrev)]), GRB.MAXIMIZE)
        self._problem.optimize()

        assignment = {}
        for paper in not_assigned:
            assignment[paper] = []
        for reviewer in range(self.numrev):
            for paper in not_assigned:
                if self._mix_vars[reviewer, paper].X == 1:
                    assignment[paper] += [reviewer]
                if np.abs(self._mix_vars[reviewer, paper].X - int(self._mix_vars[reviewer, paper].X)) > self._EPS:
                    raise ValueError('Error with rounding -- please check that demand and ability are integal')
                self._mix_vars[reviewer, paper].ub = 0
        self._problem.update()

        return assignment, current_solution


    # Join two assignments
    @staticmethod
    def _join_assignment(assignment1, assignment2):
        assignment = {}
        for paper in assignment1:
            assignment[paper] = assignment1[paper] + assignment2[paper]
        return assignment

    # Compute fairfness
    def quality(self, assignment, *args):
        qual = np.inf
        if args != ():
            paper = args[0]
            return np.sum([self.function(self.simmatrix[reviewer, paper]) for reviewer in assignment[paper]])
        else:
            for paper in assignment:
                if qual > sum([self.function(self.simmatrix[reviewer, paper]) for reviewer in assignment[paper]]):
                    qual = np.sum([self.function(self.simmatrix[reviewer, paper]) for reviewer in assignment[paper]])
        return qual
##main algo
    def _fair_assignment(self):
        
        # Counter for number of performed iterations
        iter_counter = 0
        # Start time
        start_time = time.time()
        
        current_best = None
        current_best_score = 0
        local_simmatrix = self.simmatrix.copy()
        local_abilities = self.ability
        not_assigned = set(range(self.numpapers))
        final_assignment = {}

        # one iteration of Steps 2 to 7 of the algorithm
        while not_assigned != set() and iter_counter < self.iter_limit and (time.time() < start_time + self.time_limit or iter_counter == 0):
            
            iter_counter += 1
            
            lower_bound = 0
            upper_bound = len(not_assigned) * self.numrev

            # Step 2
            for kappa in range(1, self.demand + 1):

                # Step 2(a)
                tmp_abilities = local_abilities.copy()
                tmp_simmatrix = local_simmatrix.copy()

                # Step 2(b)
                assignment1, lower_bound = self._subroutine(tmp_simmatrix, kappa, tmp_abilities, not_assigned,
                                                            lower_bound, upper_bound)

                # Step 2(c)
                for paper in assignment1:
                    for reviewer in assignment1[paper]:
                        tmp_simmatrix[reviewer, paper] = -1
                        tmp_abilities[reviewer] -= 1

                # Step 2(d)
                assignment2 = self._subroutine(tmp_simmatrix, self.demand - kappa, tmp_abilities, not_assigned,
                                               lower_bound, upper_bound)[0]

                # Step 2(e)
                assignment = self._join_assignment(assignment1, assignment2)

                # Keep track of the best candidate assignment (including the one from the prev. iteration)
                if self.quality(assignment) > current_best_score or current_best_score == 0:
                    current_best = assignment
                    current_best_score = self.quality(assignment)

            # Steps 4 to 6
            for paper in not_assigned.copy():
                # For every paper not yet fixed in the final assignment we update the assignment
                final_assignment[paper] = current_best[paper]
                # Find the most worst-off paper
                if self.quality(current_best, paper) == current_best_score:
                    # Delete it from current candidate assignment and from the set of papers which are
                    # not yet fixed in the final output
                    del current_best[paper]
                    not_assigned.discard(paper)
                    # This paper is now fixed in the final assignment

                    # Update abilities of reviewers
                    for reviewer in range(self.numrev):
                        # edges adjunct to the vertex of the most worst-off papers
                        # will not be used in the flow network any more
                        local_simmatrix[reviewer, paper] = -1
                        self._mix_vars[reviewer, paper].ub = 0
                        self._mix_vars[reviewer, paper].lb = 0
                        if reviewer in final_assignment[paper]:
                            local_abilities[reviewer] -= 1
            
            current_best_score = self.quality(current_best)
            self._problem.update()

        self.fa = final_assignment
        self.best_quality = self.quality(final_assignment)

    def fair_assignment(self):
        self._initialize_model()
        self._fair_assignment()  # same logic as your original
        # final result is in self.fa
        print(self.fa)
        print(self.best_quality)
        return self.fa
        
# Create GUI for the peer reviewer assignment
class PeerReviewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Peer Reviewer Assignment")
        self.root.geometry("800x600")
        
        self.file_path = None
        self.reviewer_capacities = []
        self.reviewer_names = []
        
        # Create frames
        self.top_frame = ttk.Frame(root, padding=10)
        self.top_frame.pack(fill=tk.X)
        
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.bottom_frame = ttk.Frame(root, padding=10)
        self.bottom_frame.pack(fill=tk.X)
        
        # File selection
        ttk.Label(self.top_frame, text="Excel File:").grid(row=0, column=0, sticky=tk.W)
        self.file_var = tk.StringVar()
        ttk.Entry(self.top_frame, textvariable=self.file_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(self.top_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        ttk.Button(self.top_frame, text="Load Reviewers", command=self.load_reviewers).grid(row=0, column=3, padx=5)
        
        # Default demand
        ttk.Label(self.top_frame, text="Reviewers per Paper:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.demand_var = tk.IntVar(value=2)
        ttk.Spinbox(self.top_frame, from_=1, to=30, textvariable=self.demand_var, width=5).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Initialize default capacity variable and modified status tracker
        self.default_capacity_var = tk.IntVar(value=8)
        self.modified_indices = set()
        ttk.Label(self.top_frame, text="Default Capacity:").grid(row=2, column=0, sticky=tk.W, pady=5)
        default_spin = ttk.Spinbox(self.top_frame, from_=1, to=100, textvariable=self.default_capacity_var, width=5)
        default_spin.grid(row=2, column=1, sticky=tk.W, pady=5)
        # Trace changes to default capacity to update unmodified reviewers
        self.default_capacity_var.trace_add('write', self.on_default_capacity_change)
        
        # Initialize preference scores mapping and management button
        self.preference_scores = DEFAULT_PREFERENCE_SCORES.copy()
        ttk.Button(self.top_frame, text="Manage Preferences", command=self.open_preferences).grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        
        # Table for reviewer capacities
        self.create_table()
        
        # Buttons
        ttk.Button(self.bottom_frame, text="Run Assignment", command=self.run_assignment).pack(side=tk.RIGHT, padx=5)
        ttk.Button(self.bottom_frame, text="Reset Capacities", command=self.reset_capacities).pack(side=tk.RIGHT, padx=5)
    
    def create_table(self):
        # Create treeview for reviewer capacities
        columns = ("Reviewer", "Capacity")
        self.tree = ttk.Treeview(self.main_frame, columns=columns, show="headings")
        
        # Define headings
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        
        # Add scrollbars
        y_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=y_scrollbar.set)
        
        # Pack elements
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click event for editing
        self.tree.bind("<Double-1>", self.edit_capacity)
        
        # Configure tag for highlighting modified capacities
        self.tree.tag_configure('modified', foreground='lightblue')
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.file_path = file_path
            self.file_var.set(file_path)
    
    def load_reviewers(self):
        if not self.file_path:
            tk.messagebox.showerror("Error", "Please select an Excel file first")
            return
        
        try:
            # Read the Excel file to get reviewer names
            df = pd.read_excel(self.file_path)
            self.reviewer_names = df.iloc[:20, 0].tolist()  # Get first 20 reviewers
            
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Reset capacities to default and clear modification flags
            default = self.default_capacity_var.get()
            self.reviewer_capacities = [default] * len(self.reviewer_names)
            self.modified_indices.clear()
            
            # Add reviewers to the table
            for i, name in enumerate(self.reviewer_names):
                self.tree.insert("", tk.END, values=(name, self.reviewer_capacities[i]))
            
            tk.messagebox.showinfo("Success", f"Loaded {len(self.reviewer_names)} reviewers")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load reviewers: {str(e)}")
    
    def edit_capacity(self, event):
        # Get the item that was clicked
        item = self.tree.identify_row(event.y)
        if not item:
            return
        
        # Get the column that was clicked
        column = self.tree.identify_column(event.x)
        if column != "#2":  # Only allow editing the capacity column
            return
        
        # Get current values
        current_values = self.tree.item(item, "values")
        reviewer_name = current_values[0]
        
        # Create a popup for editing
        popup = tk.Toplevel(self.root)
        popup.title(f"Edit Capacity for {reviewer_name}")
        popup.geometry("300x100")
        popup.transient(self.root)
        
        ttk.Label(popup, text=f"Set capacity for {reviewer_name}:").pack(pady=5)
        
        capacity_var = tk.IntVar(value=int(current_values[1]))
        capacity_spin = ttk.Spinbox(popup, from_=1, to=20, textvariable=capacity_var, width=5)
        capacity_spin.pack(pady=5)
        
        def save_capacity():
            # Update the tree view
            self.tree.item(item, values=(reviewer_name, capacity_var.get()))
            
            # Update the capacities list
            idx = self.tree.index(item)
            self.reviewer_capacities[idx] = capacity_var.get()
            
            # Mark this capacity as modified and highlight it
            self.modified_indices.add(idx)
            self.tree.item(item, tags=('modified',))
            
            popup.destroy()
        
        ttk.Button(popup, text="Save", command=save_capacity).pack(pady=5)
    
    def reset_capacities(self):
        # Reset all capacities to the default and clear modifications
        default = self.default_capacity_var.get()
        for i, item in enumerate(self.tree.get_children()):
            reviewer_name = self.tree.item(item, "values")[0]
            self.tree.item(item, values=(reviewer_name, default))
            self.reviewer_capacities[i] = default
            # Remove modified highlight
            self.tree.item(item, tags=())
        self.modified_indices.clear()
    
    def on_default_capacity_change(self, *args):
        """Update unmodified capacities when default changes"""
        default = self.default_capacity_var.get()
        for idx, item in enumerate(self.tree.get_children()):
            if idx not in self.modified_indices:
                self.tree.set(item, "Capacity", default)
                self.reviewer_capacities[idx] = default
    
    def run_assignment(self):
        if not self.file_path:
            tk.messagebox.showerror("Error", "Please select an Excel file first")
            return
        
        if not self.reviewer_capacities:
            tk.messagebox.showerror("Error", "Please load reviewers first")
            return
        
        try:
            # Get demand
            demand = self.demand_var.get()
            
            # Validate that every capacity meets the demand
            if any(cap < demand for cap in self.reviewer_capacities):
                tk.messagebox.showerror("Error", f"Reviewer capacities must be at least {demand} (Reviewers per Paper)")
                return
            
            # Run the assignment and get output filename
            res, summary_df, paper_scores_df, output_file = run_assignment(
                input_file=self.file_path,
                reviewer_capacities=self.reviewer_capacities,
                demand=demand,
                preference_scores=self.preference_scores
            )
            
            # Show success message with dynamic filename
            tk.messagebox.showinfo("Success", 
                "Assignment completed successfully!\n"
                f"Results saved to {output_file}")
            
            # Open the output file
            if os.path.exists(output_file):
                os.system(f"open \"{output_file}\"" if sys.platform == "darwin" else f"start \"{output_file}\"")
                
        except Exception as e:
            tk.messagebox.showerror("Error", f"Assignment failed: {str(e)}")

    def open_preferences(self):
        # Popup to manage preference-to-score mapping
        window = tk.Toplevel(self.root)
        window.title("Manage Preferences")
        window.geometry("400x300")
        columns = ("Preference", "Score")
        tree = ttk.Treeview(window, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        # Populate current mappings
        for label, score in self.preference_scores.items():
            tree.insert("", tk.END, values=(label, score))
        tree.pack(fill=tk.BOTH, expand=True, pady=5)
        # Buttons frame
        btn_frame = ttk.Frame(window)
        btn_frame.pack(fill=tk.X, padx=5)
        ttk.Button(btn_frame, text="Add", command=lambda: self._add_pref(tree)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Edit", command=lambda: self._edit_pref(tree)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete", command=lambda: self._delete_pref(tree)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close", command=window.destroy).pack(side=tk.RIGHT, padx=5)

    def _add_pref(self, tree):
        popup = tk.Toplevel(self.root)
        popup.title("Add Preference")
        ttk.Label(popup, text="Label:").pack(pady=5)
        label_var = tk.StringVar()
        ttk.Entry(popup, textvariable=label_var).pack()
        ttk.Label(popup, text="Score:").pack(pady=5)
        score_var = tk.IntVar(value=0)
        ttk.Spinbox(popup, from_=-1000, to=1000, textvariable=score_var).pack()
        def save():
            label = label_var.get().strip()
            score = score_var.get()
            if label:
                self.preference_scores[label] = score
                tree.insert("", tk.END, values=(label, score))
            popup.destroy()
        ttk.Button(popup, text="Save", command=save).pack(pady=5)

    def _edit_pref(self, tree):
        selected = tree.selection()
        if not selected:
            return
        item = selected[0]
        old_label, old_score = tree.item(item, "values")
        popup = tk.Toplevel(self.root)
        popup.title("Edit Preference")
        ttk.Label(popup, text="Label:").pack(pady=5)
        label_var = tk.StringVar(value=old_label)
        ttk.Entry(popup, textvariable=label_var).pack()
        ttk.Label(popup, text="Score:").pack(pady=5)
        score_var = tk.IntVar(value=int(old_score))
        ttk.Spinbox(popup, from_=-1000, to=1000, textvariable=score_var).pack()
        def save():
            new_label = label_var.get().strip()
            new_score = score_var.get()
            if new_label != old_label:
                self.preference_scores.pop(old_label, None)
            self.preference_scores[new_label] = new_score
            tree.item(item, values=(new_label, new_score))
            popup.destroy()
        ttk.Button(popup, text="Save", command=save).pack(pady=5)

    def _delete_pref(self, tree):
        selected = tree.selection()
        for item in selected:
            label = tree.item(item, "values")[0]
            self.preference_scores.pop(label, None)
            tree.delete(item)

# Import for packaging
import sys

if __name__ == "__main__":
    root = tk.Tk()
    app = PeerReviewerApp(root)
    root.mainloop()
