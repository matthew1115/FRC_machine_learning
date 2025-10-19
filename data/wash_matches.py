import csv

def wash_matches(input_filename, output_filename):
    """
    清洗比赛数据，根据三个条件过滤比赛记录：
    1. 联盟犯规罚分合计大于20分
    2. 有队伍总得分小于等于0
    3. 联盟得分调整合计绝对值大于20分
    """
    with open(input_filename, 'r', newline='', encoding='utf-8') as infile, \
         open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # 读取并写入标题行
        header = next(reader)
        writer.writerow(header)
        
        # 读取所有数据行
        rows = list(reader)
        total_rows = len(rows)
        index = 0
        
        # 处理每一场比赛（6行数据）
        while index < total_rows:
            # 获取当前比赛的6行数据
            if index + 6 > total_rows:
                break  # 不足6行，跳过
                
            match_block = rows[index:index+6]
            
            # 检查条件1：联盟犯规罚分合计是否大于20分
            fouls_alliance1 = sum(float(row[14]) for row in match_block[:3])
            fouls_alliance2 = sum(float(row[14]) for row in match_block[3:6])
            
            if fouls_alliance1 <= -15 or fouls_alliance2 <= -15:
                index += 6
                continue  # 跳过这场比赛
            
            # 检查条件2：是否有队伍总得分小于等于0
            has_invalid_score = any(float(row[16]) <= 0 for row in match_block)
            
            if has_invalid_score:
                index += 6
                continue  # 跳过这场比赛
            
            # 检查条件3：联盟得分调整合计绝对值是否大于20分
            adjustments_alliance1 = sum(float(row[15]) for row in match_block[:3])
            adjustments_alliance2 = sum(float(row[15]) for row in match_block[3:6])
            
            if abs(adjustments_alliance1) >= 15 or abs(adjustments_alliance2) >= 15:
                index += 6
                continue  # 跳过这场比赛
            
            # 如果三个条件都不满足，则保留这场比赛的数据
            for row in match_block:
                writer.writerow(row)
            
            index += 6

# 执行数据清洗
wash_matches('matches.csv', 'matches_wash.csv')
print("数据清洗完成，结果已保存到 match_wash.csv")