import csv

def wash_payoffs(matches_file, payoffs_file, output_file):
    """
    清洗playoffs.csv数据，移除其中包含未在matches.csv中出现的队伍的比赛记录
    """
    # 第一步：从matches.csv中提取所有队伍编号
    teams_in_matches = set()
    
    with open(matches_file, 'r', newline='', encoding='utf-8') as mfile:
        reader = csv.reader(mfile)
        next(reader)  # 跳过标题行
        
        for row in reader:
            if len(row) > 1:  # 确保行有足够的数据
                team_id = row[1]  # 第二列是队伍编号
                teams_in_matches.add(team_id)
    
    print(f"在matches.csv中找到 {len(teams_in_matches)} 支不同的队伍")
    
    # 第二步：处理playoffs.csv文件
    with open(payoffs_file, 'r', newline='', encoding='utf-8') as pfile, \
         open(output_file, 'w', newline='', encoding='utf-8') as ofile:
        
        reader = csv.reader(pfile)
        writer = csv.writer(ofile)
        
        # 读取并写入标题行
        header = next(reader)
        writer.writerow(header)
        
        # 读取所有数据行
        rows = list(reader)
        total_rows = len(rows)
        index = 0
        kept_matches = 0
        removed_matches = 0
        
        # 处理每一场比赛（6行数据）
        while index < total_rows:
            # 获取当前比赛的6行数据
            if index + 6 > total_rows:
                break  # 不足6行，跳过
                
            match_block = rows[index:index+6]
            
            # 检查这场比赛中的所有队伍是否都在matches.csv中出现过
            all_teams_valid = True
            for row in match_block:
                if len(row) > 1:  # 确保行有足够的数据
                    team_id = row[1]  # 第二列是队伍编号
                    if team_id not in teams_in_matches:
                        all_teams_valid = False
                        break
            
            # 如果所有队伍都有效，则保留这场比赛
            if all_teams_valid:
                for row in match_block:
                    writer.writerow(row)
                kept_matches += 1
            else:
                removed_matches += 1
            
            index += 6
    
    print(f"处理完成: 保留了 {kept_matches} 场比赛，移除了 {removed_matches} 场比赛")
    print(f"结果已保存到 {output_file}")

# 执行数据清洗
wash_payoffs('matches.csv', 'playoffs.csv', 'playoffs_exist.csv')