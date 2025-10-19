import requests
from bs4 import BeautifulSoup
import csv
import re
from datetime import datetime
from urllib.parse import urljoin

def get_first_parentheses(text):
    match = re.search(r'\((.*?)\)', text)
    return match.group(1) if match else "0"
	
def find_first_index(lst, target):
    return next((i for i, s in enumerate(lst) if s == target), -1)

def extract_detailed_result_table(url):
    """
    从指定URL提取Detailed Result下的match-table表格数据
    返回表格数据（所有行数据）
    """
    try:
        # 设置请求头模拟浏览器访问
        # "User-Agent":"MyResearchBot/1.0"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Referer": "https://www.thebluealliance.com/"
        }
        
        print(f"正在访问页面: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        
        #print("解析HTML内容...")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 获取页面标题用于文件名
        page_title = "detailed_result"
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text(strip=True).rpartition('-')[0].strip()
            # 清理标题用于文件名
            page_title = re.sub(r'[\\/*?:"<>|]', '', title_text)[:50]
            #print(f"页面标题: {title_text}")
        
        # 查找match-breakdown div
        breakdown_div = soup.find('div', {'id': 'match-breakdown'})
        if not breakdown_div:
            raise ValueError("未找到id为'match-breakdown'的div")
        
        # 查找Detailed Result标题
        detailed_result_h3 = None
        for h3 in breakdown_div.find_all('h3'):
            if 'Detailed Result' in h3.get_text():
                detailed_result_h3 = h3
                break
        
        if not detailed_result_h3:
            raise ValueError("未找到'Detailed Result'标题")
        
        #print("找到Detailed Result标题")
        
        # 查找紧随其后的match-table表格
        table = detailed_result_h3.find_next('table', class_='match-table')
        if not table:
            raise ValueError("在Detailed Result标题后未找到class为'match-table'的表格")
        
        #print("找到Detailed Result下的match-table表格")
        
        # 提取表格数据 - 所有行
        data = []
        rows = table.find_all('tr')
        header = ["Match"]
        header.extend([title_text] * 6)
        #print(header)
        data.append(header)
        level_prefix = ""
        team_weight = [0.33,0.33,0.33]
        endgame_score_left = []
        endgame_score_right = []
		
        for row in rows:
            cells = []
            auto_leave_score = []

            for cell in row.find_all(['td', 'th']):
                # 处理多个span标签的情况
                span_contents = []
                
                # 1. 首先检查带有rel属性的span标签
                rel_spans = cell.find_all('span', rel=True)
                for span in rel_spans:
                    # 2. 如果有带rel属性的span，只使用它们的title
                    if span.get('title'):
                        span_contents.append(span.get('title').strip())
                        cells.append(span.get('title').strip())
                        classes = span.get('class', [])
                        # 检查是否包含目标class
                        if 'glyphicon-ok' in classes:
                            auto_leave_score.append(3)
                        else:
                            auto_leave_score.append(0)

                # 3. 否则使用单元格的文本内容
                cell_text = cell.get_text(strip=True)
                cells.append(cell_text.replace('\r', '').replace('\n', ''))

            if cells:  # 忽略空行
                # print(f"cells:{cells}")
                formatted_cells = cells[:]
                
                # Auto Leave				
                index = find_first_index(cells,"Auto Leave")
                if index != -1:
                    del formatted_cells[8]
                    del formatted_cells[index]
                    del formatted_cells[3]
                    formatted_cells.insert(0,"Teams")
                    #print(f"Formatted cells:{formatted_cells}")
                    data.append(formatted_cells)
					
                    formatted_cells = []
                    formatted_cells.append(cells[index])
                    formatted_cells.extend(map(str,auto_leave_score))

                # Auto Coral 
                index = find_first_index(cells,"Auto Coral Count")
                if index != -1:
                    level_prefix = "Auto Coral"

                # Auto Algae
                if any(('Auto' in s) and ('Algae' in s) for s in cells):
                    print("Auto Algae Missing")

                # Auto Coral Points - skip
                index = find_first_index(cells,"Auto Coral Points")
                if index != -1:
                    continue
                # Total Auto points - skip
                index = find_first_index(cells,"Total Auto")
                if index != -1:
                    continue

                # Teleop Coral
                index = find_first_index(cells,"Teleop Coral Count")
                if index != -1:
                    level_prefix = "Teleop Coral"

                # Teleop Coral Points - skip
                index = find_first_index(cells,"Teleop Coral Points")
                if index != -1:
                    continue

                
                if cells[0].startswith("L"):
                    if level_prefix == "Auto Coral":
                        # Auto L4-L1 each get 7,6,4,3
                        formatted_cells = []
                        formatted_cells.append("Auto Coral " + cells[0])
                        if cells[0].startswith("L4"):
                            formatted_cells.append(f"{float(cells[1]) * team_weight[0] * 7:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[1] * 7:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[2] * 7:.1f}")
                            formatted_cells.append(f"{float(cells[3]) * team_weight[0] * 7:.1f}")
                            formatted_cells.append(f"{float(cells[3]) * team_weight[1] * 7:.1f}")
                            formatted_cells.append(f"{float(cells[3]) * team_weight[2] * 7:.1f}")
                        if cells[0].startswith("L3"):
                            formatted_cells.append(f"{float(cells[1]) * team_weight[0] * 6:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[1] * 6:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[2] * 6:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[0] * 6:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[1] * 6:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[2] * 6:.1f}")
                        if cells[0].startswith("L2"):
                            formatted_cells.append(f"{float(cells[1]) * team_weight[0] * 4:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[1] * 4:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[2] * 4:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[0] * 4:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[1] * 4:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[2] * 4:.1f}")
                        if cells[0].startswith("L1"):
                            formatted_cells.append(f"{float(cells[1]) * team_weight[0] * 3:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[1] * 3:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[2] * 3:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[0] * 3:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[1] * 3:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[2] * 3:.1f}")
                    elif level_prefix == "Teleop Coral":
                        # Teleop L4-L1 each get 5,4,3,2
                        formatted_cells = []
                        formatted_cells.append("Teleop Coral " + cells[0])
                        if cells[0].startswith("L4"):
                            formatted_cells.append(f"{float(cells[1]) * team_weight[0] * 5:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[1] * 5:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[2] * 5:.1f}")
                            formatted_cells.append(f"{float(cells[3]) * team_weight[0] * 5:.1f}")
                            formatted_cells.append(f"{float(cells[3]) * team_weight[1] * 5:.1f}")
                            formatted_cells.append(f"{float(cells[3]) * team_weight[2] * 5:.1f}")
                        if cells[0].startswith("L3"):
                            formatted_cells.append(f"{float(cells[1]) * team_weight[0] * 4:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[1] * 4:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[2] * 4:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[0] * 4:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[1] * 4:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[2] * 4:.1f}")
                        if cells[0].startswith("L2"):
                            formatted_cells.append(f"{float(cells[1]) * team_weight[0] * 3:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[1] * 3:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[2] * 3:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[0] * 3:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[1] * 3:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[2] * 3:.1f}")
                        if cells[0].startswith("L1"):
                            formatted_cells.append(f"{float(cells[1]) * team_weight[0] * 2:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[1] * 2:.1f}")
                            formatted_cells.append(f"{float(cells[1]) * team_weight[2] * 2:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[0] * 2:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[1] * 2:.1f}")
                            formatted_cells.append(f"{float(cells[2]) * team_weight[2] * 2:.1f}")

                # Processor Algae Count - each get 6 points
                index = find_first_index(cells,"Processor Algae Count")
                if index != -1:
                    formatted_cells = []
                    formatted_cells.append("Processor Algae")
                    try:
                        formatted_cells.append(f"{float(cells[0]) * team_weight[0] * 6:.1f}")
                        formatted_cells.append(f"{float(cells[0]) * team_weight[1] * 6:.1f}")
                        formatted_cells.append(f"{float(cells[0]) * team_weight[2] * 6:.1f}")
                    except ValueError:
                        print("Float type convert error!")
                        formatted_cells.append("0")
                        formatted_cells.append("0")
                        formatted_cells.append("0")
                    try:
                        formatted_cells.append(f"{float(cells[2]) * team_weight[0] * 6:.1f}")
                        formatted_cells.append(f"{float(cells[2]) * team_weight[1] * 6:.1f}")
                        formatted_cells.append(f"{float(cells[2]) * team_weight[2] * 6:.1f}")
                    except ValueError:
                        print("Float type convert error!")
                        formatted_cells.append("0")
                        formatted_cells.append("0")
                        formatted_cells.append("0")
					
                # Net Algae Count - each get 4 points
                index = find_first_index(cells,"Net Algae Count")
                if index != -1:
                    formatted_cells = []
                    formatted_cells.append("Net Algae")
                    try:
                        formatted_cells.append(f"{float(cells[0]) * team_weight[0] * 4:.1f}")
                        formatted_cells.append(f"{float(cells[0]) * team_weight[1] * 4:.1f}")
                        formatted_cells.append(f"{float(cells[0]) * team_weight[2] * 4:.1f}")
                    except ValueError:
                        print("Float type convert error!")
                        formatted_cells.append("0")
                        formatted_cells.append("0")
                        formatted_cells.append("0")
                    try:
                        formatted_cells.append(f"{float(cells[2]) * team_weight[0] * 4:.1f}")
                        formatted_cells.append(f"{float(cells[2]) * team_weight[1] * 4:.1f}")
                        formatted_cells.append(f"{float(cells[2]) * team_weight[2] * 4:.1f}")
                    except ValueError:
                        print("Float type convert error!")
                        formatted_cells.append("0")
                        formatted_cells.append("0")
                        formatted_cells.append("0")
					
                # Algae Points - skip
                index = find_first_index(cells,"Algae Points")
                if index != -1:
                    continue

                # Endgame - 3 lines, compress to 1 row
                index = find_first_index(cells,"Robot 1 Endgame")
                if index != -1:
                    endgame_score_left.append(get_first_parentheses(cells[1]).replace('+',''))
                    endgame_score_right.append(get_first_parentheses(cells[4]).replace('+',''))
                    continue
					
                index = find_first_index(cells,"Robot 2 Endgame")
                if index != -1:
                    endgame_score_left.append(get_first_parentheses(cells[1]).replace('+',''))
                    endgame_score_right.append(get_first_parentheses(cells[4]).replace('+',''))
                    continue

                index = find_first_index(cells,"Robot 3 Endgame")
                if index != -1:
                    endgame_score_left.append(get_first_parentheses(cells[1]).replace('+',''))
                    endgame_score_right.append(get_first_parentheses(cells[4]).replace('+',''))
                    formatted_cells = []
                    formatted_cells.append("Endgame")
                    formatted_cells.extend(endgame_score_left)
                    formatted_cells.extend(endgame_score_right)
                    
                # Barge Points - skip
                index = find_first_index(cells,"Barge Points")
                if index != -1:
                    continue

                # Total Teleop - skip
                index = find_first_index(cells,"Total Teleop")
                if index != -1:
                    continue

                # Coopertition Criteria Met - skip
                index = find_first_index(cells,"Coopertition Criteria Met")
                if index != -1:
                    continue

                # Auto Bonus - skip
                index = find_first_index(cells,"Auto Bonus")
                if index != -1:
                    continue

                # Coral Bonus - skip
                index = find_first_index(cells,"Coral Bonus")
                if index != -1:
                    continue

                # Barge Bonus - skip
                index = find_first_index(cells,"Barge Bonus")
                if index != -1:
                    continue

                # Fouls / Major Fouls - skips
                index = find_first_index(cells,"Fouls / Major Fouls")
                if index != -1:
                    continue

                # Foul Points
                index = find_first_index(cells,"Foul Points")
                if index != -1:
                    formatted_cells = []
                    formatted_cells.append("Fouls")
                    try:
                        formatted_cells.append(f"{float(cells[2]) * -1/3:.1f}")
                        formatted_cells.append(f"{float(cells[2]) * -1/3:.1f}")
                        formatted_cells.append(f"{float(cells[2]) * -1/3:.1f}")
                    except ValueError:
                        print("Float type convert error!")
                        formatted_cells.append("0")
                        formatted_cells.append("0")
                        formatted_cells.append("0")
                    try:
                        formatted_cells.append(f"{float(cells[0]) * -1/3:.1f}")
                        formatted_cells.append(f"{float(cells[0]) * -1/3:.1f}")
                        formatted_cells.append(f"{float(cells[0]) * -1/3:.1f}")
                    except ValueError:
                        print("Float type convert error!")
                        formatted_cells.append("0")
                        formatted_cells.append("0")
                        formatted_cells.append("0")

                # Adjustments
                index = find_first_index(cells,"Adjustments")
                if index != -1:
                    formatted_cells = []
                    formatted_cells.append("Adjustments")
                    try:
                        formatted_cells.append(f"{float(cells[0]) * team_weight[0]:.1f}")
                        formatted_cells.append(f"{float(cells[0]) * team_weight[1]:.1f}")
                        formatted_cells.append(f"{float(cells[0]) * team_weight[2]:.1f}")
                    except ValueError:
                        print("Float type convert error!")
                        formatted_cells.append("0")
                        formatted_cells.append("0")
                        formatted_cells.append("0")
                    try:
                        formatted_cells.append(f"{float(cells[2]) * team_weight[0]:.1f}")
                        formatted_cells.append(f"{float(cells[2]) * team_weight[1]:.1f}")
                        formatted_cells.append(f"{float(cells[2]) * team_weight[2]:.1f}")
                    except ValueError:
                        print("Float type convert error!")
                        formatted_cells.append("0")
                        formatted_cells.append("0")
                        formatted_cells.append("0")
						
                # Total Score - skip
                index = find_first_index(cells,"Total Score")
                if index != -1:
                    continue

                # Ranking Points - skip
                index = find_first_index(cells,"Ranking Points")
                if index != -1:
                    continue


                # print(f"Formatted cells:{formatted_cells}")
                data.append(formatted_cells)
            
        # calculate total score for each team        
        num_rows = len(data)
        num_cols = len(data[1])
        total_score = [0] * (num_cols - 1)
        formatted_total_score = ["0"] * (num_cols - 1)
        # 是否包含比赛信息
        if data[0][0] == "Match":
            k = 2
        else:
            k = 1		
        for i in range(k,num_rows):
            data_row = data[i]
            #print(f"data_row:{data_row}")
            for j in range(1,num_cols):
                total_score[j-1] += float(data_row[j])      
        for j in range(0,num_cols - 1):
            formatted_total_score[j] = f"{total_score[j]:.1f}" 		
        #print(f"total_score:{formatted_total_score}")    
        data.append(["Total Score"]+formatted_total_score)
        #print(data[num_rows])

        # judge win,loss or draw
        if (total_score[0]+total_score[1]+total_score[2])-(total_score[3]+total_score[4]+total_score[5])>0.001:
            data.append(["Result"]+["Win"]+["Win"]+["Win"]+["Loss"]+["Loss"]+["Loss"])
        elif (total_score[0]+total_score[1]+total_score[2])-(total_score[3]+total_score[4]+total_score[5])<-0.001:
            data.append(["Result"]+["Loss"]+["Loss"]+["Loss"]+["Win"]+["Win"]+["Win"])
        else:
            data.append(["Result"]+["Draw"]+["Draw"]+["Draw"]+["Draw"]+["Draw"]+["Draw"])
        return [list(data_row_T) for data_row_T in zip(*data)], page_title
    
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
    
    return [], "detailed_result"

def extract_qual_match_urls(url):
    """
    从指定URL提取qual-match-table中以"Quals"开头的<a>标签的URL
    返回URL列表
    """
    try:
        # 设置请求头模拟浏览器访问
        # "User-Agent":"MyResearchBot/1.0"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Referer": "https://www.thebluealliance.com/"
        }
        
        print(f"正在访问页面: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        
        print("解析HTML内容...")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找qual-match-table表格
        qual_table = soup.find('table', {'id': 'qual-match-table'})
        if not qual_table:
            raise ValueError("未找到id为'qual-match-table'的表格")
        
        print("找到qual-match-table表格")
        
        # 提取所有class为visible-lg的tr标签
        visible_lg_rows = qual_table.find_all('tr', class_='visible-lg')
        if not visible_lg_rows:
            raise ValueError("未找到class为'visible-lg'的tr标签")
        
        print(f"找到 {len(visible_lg_rows)} 个visible-lg的tr标签")
        
        # 提取这些tr中以"Quals"开头的<a>标签的URL
        qual_links = []
        for row in visible_lg_rows:
            for a_tag in row.find_all('a'):
                link_text = a_tag.get_text(strip=True)
                # 检查链接文本是否以"Quals"开头
                if link_text.startswith('Quals'):
                    href = a_tag.get('href')
                    if href:
                        # 将相对URL转换为绝对URL
                        full_url = urljoin('https://www.thebluealliance.com/', href)
                        qual_links.append(full_url)
        
        print(f"找到 {len(qual_links)} 个Quals开头的链接")
        return qual_links
    
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
    
    return []

def extract_2025_event_urls():
    """
    列表页面提取所有2025年赛事链接
    返回2025年赛事URL列表
    """
    base_url = "https://www.thebluealliance.com/events#"
    event_prefix = "https://www.thebluealliance.com/event/2025"
    
    try:
        # 设置请求头模拟浏览器访问
        # "User-Agent":"MyResearchBot/1.0"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Referer": "https://www.thebluealliance.com/"
        }
        
        print(f"正在访问赛事列表页面: {base_url}")
        response = requests.get(base_url, headers=headers, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        
        print("解析HTML内容...")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 提取所有<a>标签
        all_links = soup.find_all('a')
        print(f"找到 {len(all_links)} 个链接")
        
        # 提取符合条件的赛事链接
        event_links = set()  # 使用set自动去重
        for link in all_links:
            href = link.get('href')
            if href:
                # 将相对URL转换为绝对URL
                full_url = urljoin('https://www.thebluealliance.com/', href)
                
                # 检查URL是否符合2025年赛事格式
                if full_url.startswith(event_prefix):
                    # 验证URL格式：/event/2025后跟字母数字和短划线
                    if re.match(r'https://www\.thebluealliance\.com/event/2025[a-zA-Z0-9\-]+$', full_url):
                        event_links.add(full_url)
        
        print(f"找到 {len(event_links)} 个2025年赛事链接")
        return sorted(list(event_links))  # 排序后返回
    
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
    
    return []

def print_data_preview(data, max_rows=20):
    """打印数据预览"""
    if not data:
        print("没有数据可预览")
        return
    
    print("\n数据预览:")
    
    # 打印数据行
    for i, row in enumerate(data[:max_rows]):
        # 为长内容添加换行处理
        formatted_row = []
        for cell in row:
            if len(cell) > 50:
                formatted_row.append(cell[:47] + "...")
            else:
                formatted_row.append(cell)
        print(" | ".join(formatted_row))
        # print(formatted_row)
    if len(data) > max_rows:
        print(f"... 显示前 {max_rows}/{len(data)} 行 ...")

def main():
    """
    目标URL
    "https://www.thebluealliance.com/robots.txt" 内容：
        User-agent: *
        Allow: /
        Disallow: /_
        Disallow: /advanced_search?*
        Disallow: /admin/
        Disallow: /account/
        Disallow: /api/
        Disallow: /backend-tasks/
        Disallow: /request/
        Disallow: /suggest/
        Disallow: /tasks/
        Disallow: /mytba/
        Disallow: /nearby?*
    """
    #event_links = extract_2025_event_urls()
    qual_links = []
    # print(f"event_links数量:{len(event_links)}")
    url = "https://www.thebluealliance.com/event/2025cnsh"
    # 提取资格赛链接
    print(f"event_url:{url}")
    qual_links.extend(extract_qual_match_urls(url))
    print(f"qual_links数量:{len(qual_links)}")
	
    if qual_links:
        # 创建文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"FRC2025_QUALS_SH_{timestamp}.csv"

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                csv_title = 0
                for qm_url in qual_links:
                    # 提取表格数据
                    data, page_title = extract_detailed_result_table(qm_url)
                    if data:
                        # 写入所有数据行
                        if csv_title == 0:
                            csv_title = 1
                        else:
                            del data[0]
                        writer.writerows(data)
                    else:
                        print(f"{url}未提取到有效数据")				
            print(f"详细结果已保存为 {filename} ({len(qual_links)} 场数据)")
        except Exception as e:
            print(f"保存CSV文件时出错: {e}")
    else:
        print("未提取到有效链接")

if __name__ == "__main__":
    main()