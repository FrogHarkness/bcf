import pandas as pd
import numpy as np
df = pd.read_excel("EVO.xlsx")

df = df[:20]
df = df.drop(columns=['Reviewers'])


M = df.shape[1]  # Number of papers
N = df.shape[0]  # Number of reviewers

simmatrix = np.zeros((M, N), dtype=float)


PREFERENCE_SCORES = {
    "Conflict of Interest": -100,
    "None": -100,
    None: -100,
    "Moderate": 1,
    "Moderate*": 1,
    "Moderate**": 1,
    "Considerable": 5,
    "Considerable*": 5,
    "Considerable**": 5
}

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
simmatrix = np.transpose(simmatrix)##always transpose the matrix



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

def find_minimum_reviewer_capacity(simmatrix, demand=2, initial_max=3, max_iterations=100):
    """
    Find the minimum capacity for each reviewer to support a valid assignment.
    
    This function uses a more robust approach to adjust reviewer capacities until
    a valid assignment is found, then refines to find the true minimum.
    """
    numrev = simmatrix.shape[0]
    numpapers = simmatrix.shape[1]
    
    # Initialize all reviewers with the initial_max capacity
    capacities = np.ones(numrev, dtype=int) * initial_max
    
    # Calculate expertise score and workload potential for each reviewer
    expertise_scores = np.zeros(numrev)
    potential_papers = np.zeros(numrev) 
    
    for r in range(numrev):
        # Count number of papers with non-negative compatibility
        valid_papers = 0
        total_expertise = 0
        for p in range(numpapers):
            if simmatrix[r, p] >= 0:
                valid_papers += 1
                total_expertise += simmatrix[r, p]
        
        expertise_scores[r] = total_expertise
        potential_papers[r] = valid_papers
    
    # Normalize expertise scores
    if np.sum(expertise_scores) > 0:
        expertise_scores = expertise_scores / np.sum(expertise_scores)
    
    # Calculate theoretical minimum total assignments needed
    total_assignments_needed = numpapers * demand
    
    # Set minimum initial capacities based on theoretical load distribution
    # Ensure reviewers with more expertise/potential get proportionally higher min capacity
    if np.sum(potential_papers) > 0:
        adjusted_potential = potential_papers / np.sum(potential_papers)
        for r in range(numrev):
            theoretical_load = int(np.ceil(adjusted_potential[r] * total_assignments_needed))
            if theoretical_load > capacities[r]:
                capacities[r] = theoretical_load
    
    # Track failed attempts to avoid repeating
    failed_capacity_configurations = []
    
    iteration = 0
    assignment_found = False
    last_exception = None
    
    while not assignment_found and iteration < max_iterations:
        iteration += 1
        
        # Skip if we've already tried this configuration
        capacity_config = tuple(capacities)
        if capacity_config in failed_capacity_configurations:
            # Increase all capacities by 1 to break out of repeating pattern
            capacities = capacities + 1
            continue
            
        try:
            # Try to find an assignment with current capacities
            assigner = auto_assigner(
                simmatrix=simmatrix,
                demand=demand,           
                ability=capacities,
                function=lambda x: x,
                iter_limit=900000,
                time_limit=9000
            )
            
            assignment = assigner.fair_assignment()
            assignment_found = True
            
            # Calculate actual usage
            used_capacity = np.zeros(numrev, dtype=int)
            for paper, reviewers in assignment.items():
                for reviewer in reviewers:
                    used_capacity[reviewer] += 1
            
            # Refine capacities: reduce to actual usage where possible
            for r in range(numrev):
                if used_capacity[r] > 0:  # Don't reduce to zero
                    capacities[r] = max(used_capacity[r], 1)
                    
            # Check if this is the minimum possible configuration
            min_test_capacities = capacities.copy()
            
            # Iteratively try to reduce capacities further one by one
            further_reduction = True
            while further_reduction:
                further_reduction = False
                for r in range(numrev):
                    if min_test_capacities[r] > used_capacity[r] and min_test_capacities[r] > 1:
                        test_capacities = min_test_capacities.copy()
                        test_capacities[r] -= 1
                        
                        try:
                            test_assigner = auto_assigner(
                                simmatrix=simmatrix,
                                demand=demand,
                                ability=test_capacities,
                                function=lambda x: x,
                                iter_limit=10000,
                                time_limit=30
                            )
                            
                            test_assignment = test_assigner.fair_assignment()
                            min_test_capacities[r] -= 1
                            further_reduction = True
                            
                        except Exception:
                            # Can't reduce this reviewer's capacity further
                            pass
                
            capacities = min_test_capacities
            
            # Run one more time with final capacities to get the optimized assignment
            final_assigner = auto_assigner(
                simmatrix=simmatrix,
                demand=demand,
                ability=capacities,
                function=lambda x: x,
                iter_limit=900000,
                time_limit=9000
            )
            
            final_assignment = final_assigner.fair_assignment()
            
            # Recalculate actual usage
            final_used_capacity = np.zeros(numrev, dtype=int)
            for paper, reviewers in final_assignment.items():
                for reviewer in reviewers:
                    final_used_capacity[reviewer] += 1
                    
            return capacities, final_used_capacity, final_assignment
                
        except Exception as e:
            last_exception = e
            # Assignment failed with current capacities
            print(f"Assignment failed on iteration {iteration}, adjusting capacities...")
            failed_capacity_configurations.append(capacity_config)
            
            # More sophisticated capacity increase strategy:
            # 1. Identify bottleneck reviewers (those with high expertise but low capacity)
            # 2. Identify papers with fewer qualified reviewers
            # 3. Increase capacity more for reviewers that can handle difficult papers
            
            # Count qualified reviewers per paper
            qualified_reviewers_per_paper = np.zeros(numpapers)
            for p in range(numpapers):
                qualified_reviewers_per_paper[p] = sum(1 for r in range(numrev) if simmatrix[r, p] >= 0)
            
            # Find papers with fewest qualified reviewers
            difficult_papers = np.argsort(qualified_reviewers_per_paper)[:max(1, numpapers // 4)]
            
            # Count how many difficult papers each reviewer is qualified for
            difficult_paper_coverage = np.zeros(numrev)
            for r in range(numrev):
                difficult_paper_coverage[r] = sum(1 for p in difficult_papers if simmatrix[r, p] >= 0)
            
            # Combine expertise scores and difficult paper coverage to prioritize reviewers
            combined_score = 0.6 * expertise_scores + 0.4 * (difficult_paper_coverage / max(1, np.max(difficult_paper_coverage)))
            
            # Prioritize reviewers with high combined scores
            reviewers_to_increase = sorted(range(numrev), key=lambda r: combined_score[r], reverse=True)
            
            # Increase capacity more intelligently:
            # - Top 10% get +2 capacity
            # - Next 20% get +1 capacity
            top_10_percent = max(1, int(numrev * 0.1))
            next_20_percent = max(1, int(numrev * 0.2))
            
            for i in range(top_10_percent):
                capacities[reviewers_to_increase[i]] += 2
                
            for i in range(top_10_percent, top_10_percent + next_20_percent):
                if i < len(reviewers_to_increase):
                    capacities[reviewers_to_increase[i]] += 1
    
    if assignment_found:
        # Calculate actual usage
        used_capacity = np.zeros(numrev, dtype=int)
        for paper, reviewers in assignment.items():
            for reviewer in reviewers:
                used_capacity[reviewer] += 1
                
        return capacities, used_capacity, assignment
    else:
        print(f"Could not find a valid assignment after {max_iterations} iterations")
        print(f"Last error: {last_exception}")
        return None, None, None

# Find minimum reviewer capacity
print("Finding minimum reviewer capacity...")
min_capacities, actual_usage, optimized_assignment = find_minimum_reviewer_capacity(
    simmatrix,
    demand=2,
    initial_max=3
)

if min_capacities is not None:
    print("Minimum capacities found!")
    
    # Create detailed capacity report
    withnameoriginaldf = pd.read_excel("EVO.xlsx")
    capacity_report = pd.DataFrame({
        "Reviewer": [withnameoriginaldf.iloc[r, 0] for r in range(len(min_capacities))],
        "Minimum Capacity": min_capacities,
        "Actual Usage": actual_usage
    })
    
    # Calculate the number of papers assigned to each reviewer
    reviewer_counts = {}
    for paper, reviewers in optimized_assignment.items():
        for reviewer in reviewers:
            if reviewer in reviewer_counts:
                reviewer_counts[reviewer] += 1
            else:
                reviewer_counts[reviewer] = 1
    
    # Find 4 backup reviewers for each paper
    backup_reviewers = {}
    for paper_idx in range(len(df.columns)):
        if paper_idx in optimized_assignment:
            # Get the current assigned reviewers
            assigned_reviewers = optimized_assignment[paper_idx]
            
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
    
    # Create the main assignment dataframe
    ndf = pd.DataFrame(columns=df.columns)
    for col_idx, reviewers in optimized_assignment.items():
        if col_idx < len(df.columns):
            col_name = df.columns[col_idx]
            for i, reviewer in enumerate(reviewers):
                ndf.at[i, col_name] = df.iloc[reviewer, col_idx]
    
    # Create the dataframe with reviewer names
    nameevodf = pd.DataFrame(columns=df.columns)
    for col_idx, reviewers in optimized_assignment.items():
        if col_idx < len(df.columns):
            col_name = df.columns[col_idx]
            for i, reviewer in enumerate(reviewers):
                nameevodf.at[i, col_name] = withnameoriginaldf.iloc[reviewer, 0]
    
    # Add backup reviewers
    for paper_idx, backups in backup_reviewers.items():
        if paper_idx < len(df.columns):
            col_name = df.columns[paper_idx]
            for i, (rev_idx, score, count) in enumerate(backups):
                nameevodf.at[i+2, col_name] = f"{withnameoriginaldf.iloc[rev_idx, 0]} ({count})"
                ndf.at[i+2, col_name] = f"{df.iloc[rev_idx, paper_idx]} ({count})"
    
    # Create a summary table of reviewer assignments
    summary_df = pd.DataFrame(columns=["Reviewer", "Number of Assignments", "Minimum Capacity", "Capacity Utilization (%)"])
    for reviewer, count in reviewer_counts.items():
        row_idx = reviewer
        summary_df.at[row_idx, "Reviewer"] = withnameoriginaldf.iloc[reviewer, 0]
        summary_df.at[row_idx, "Number of Assignments"] = count
        summary_df.at[row_idx, "Minimum Capacity"] = min_capacities[reviewer]
        summary_df.at[row_idx, "Capacity Utilization (%)"] = (count / min_capacities[reviewer]) * 100 if min_capacities[reviewer] > 0 else 0
    
    # Sort by number of assignments (descending)
    summary_df = summary_df.sort_values(by="Number of Assignments", ascending=False).reset_index(drop=True)
    
    # Create a writer to save multiple sheets
    with pd.ExcelWriter('optimized_result.xlsx') as writer:
        nameevodf.to_excel(writer, sheet_name='Reviewer Assignments')
        ndf.to_excel(writer, sheet_name='Raw Assignments')
        summary_df.to_excel(writer, sheet_name='Assignment Summary')
        capacity_report.to_excel(writer, sheet_name='Capacity Report')
    
    print("Results saved to optimized_result.xlsx")
else:
    print("Failed to find minimum capacities")

# Original assignment algorithm (kept for comparison)
res = auto_assigner(simmatrix=simmatrix,
    demand=2,           # e.g., each paper needs exactly 2 reviewers
    ability=8,          # e.g., each reviewer can review up to 3 papers
    function=lambda x: x,   # identity: we want to sum raw scores
    iter_limit=900000,      # or whatever you need
    time_limit=9000      # 5-minute time limit, for instance
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

maxkeys = max(res.items())
maxkeys

# Create the main assignment dataframe
ndf = pd.DataFrame(columns=df.columns)
for col_idx, (row1, row2) in res.items():
    if col_idx < len(df.columns):
        col_name = df.columns[col_idx]
        ndf.at[0, col_name] = df.iloc[row1, col_idx]
        ndf.at[1, col_name] = df.iloc[row2, col_idx]

# Create the dataframe with reviewer names
withnameoriginaldf = pd.read_excel("EVO.xlsx")
nameevodf = pd.DataFrame(columns=df.columns)
for col_idx, (row1, row2) in res.items():
    if col_idx < len(df.columns):
        col_name = df.columns[col_idx]
        nameevodf.at[0, col_name] = withnameoriginaldf.iloc[row1, 0]
        nameevodf.at[1, col_name] = withnameoriginaldf.iloc[row2, 0]

# Add backup reviewers
for paper_idx, backups in backup_reviewers.items():
    if paper_idx < len(df.columns):
        col_name = df.columns[paper_idx]
        for i, (rev_idx, score, count) in enumerate(backups):
            nameevodf.at[i+2, col_name] = f"{withnameoriginaldf.iloc[rev_idx, 0]} ({count})"
            ndf.at[i+2, col_name] = f"{df.iloc[rev_idx, paper_idx]} ({count})"

# Create a summary table of reviewer assignments
summary_df = pd.DataFrame(columns=["Reviewer", "Number of Assignments"])
row_idx = 0
for reviewer, count in reviewer_counts.items():
    summary_df.at[row_idx, "Reviewer"] = withnameoriginaldf.iloc[reviewer, 0]
    summary_df.at[row_idx, "Number of Assignments"] = count
    row_idx += 1

# Sort by number of assignments (descending)
summary_df = summary_df.sort_values(by="Number of Assignments", ascending=False).reset_index(drop=True)

# Create a writer to save multiple sheets
with pd.ExcelWriter('final_result.xlsx') as writer:
    nameevodf.to_excel(writer, sheet_name='Reviewer Assignments')
    ndf.to_excel(writer, sheet_name='Raw Assignments')
    summary_df.to_excel(writer, sheet_name='Assignment Summary')