import csv

#调整matches.csv数据，使用胜率加权分配各单项得分

# Step 1: Calculate win rates for all teams
team_stats = {}  # key: team number, value: {'total': int, 'win': int}

with open('matches.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # skip header
    for row in reader:
        team = row[1]  # second column, teams
        result = row[17]  # 18th column, result

        if team not in team_stats:
            team_stats[team] = {'total': 0, 'win': 0}
        
        team_stats[team]['total'] += 1
        if result == 'Win':
            team_stats[team]['win'] += 1

# Calculate win rate for each team
team_win_rates = {}
for team, stats in team_stats.items():
    total = stats['total']
    win = stats['win']
    if total > 0:
        win_rate = win / total
    else:
        win_rate = 0.0
    team_win_rates[team] = win_rate

# Step 2: Process the file again and adjust scores
score_cols = [3,4,5,6,7,8,9,10,11,12,14,15]  # indices of score columns
total_score_col = 16  # index of Total Score column (Q列)

with open('matches.csv', 'r') as infile, open('matches_weight.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    header = next(reader)
    writer.writerow(header)  # write header

    # Read 6 rows at a time
    while True:
        rows = []
        for i in range(6):
            try:
                row = next(reader)
            except StopIteration:
                break
            rows.append(row)
        if len(rows) < 6:
            break

        # Process alliance A (first 3 rows) and alliance B (last 3 rows)
        allianceA = rows[0:3]
        allianceB = rows[3:6]

        # For alliance A
        teamsA = [row[1] for row in allianceA]
        win_ratesA = [team_win_rates.get(team, 0.0) for team in teamsA]
        total_win_rateA = sum(win_ratesA)
        if abs(total_win_rateA) < 1e-6:
            weightsA = [1/3] * 3
        else:
            weightsA = [w / total_win_rateA for w in win_ratesA]

        for col in score_cols:
            total_score = 0.0
            for row in allianceA:
                try:
                    total_score += float(row[col])
                except ValueError:
                    total_score += 0.0
            for i in range(3):
                new_value = total_score * weightsA[i]
                allianceA[i][col] = format(round(new_value, 2), '.2f')

        # For alliance B
        teamsB = [row[1] for row in allianceB]
        win_ratesB = [team_win_rates.get(team, 0.0) for team in teamsB]
        total_win_rateB = sum(win_ratesB)
        if abs(total_win_rateB) < 1e-6:
            weightsB = [1/3] * 3
        else:
            weightsB = [w / total_win_rateB for w in win_ratesB]

        for col in score_cols:
            total_score = 0.0
            for row in allianceB:
                try:
                    total_score += float(row[col])
                except ValueError:
                    total_score += 0.0
            for i in range(3):
                new_value = total_score * weightsB[i]
                allianceB[i][col] = format(round(new_value, 2), '.2f')

        # Recalculate total scores for all rows
        for row in allianceA + allianceB:
            total = 0.0
            # Sum columns 2 to 15 (indices 2 to 15)
            for col in range(2, 16):
                try:
                    total += float(row[col])
                except ValueError:
                    total += 0.0
            row[total_score_col] = format(round(total, 2), '.2f')

        new_rows = allianceA + allianceB
        for row in new_rows:
            writer.writerow(row)