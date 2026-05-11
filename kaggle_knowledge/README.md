# Kaggle竞赛数据采集工具

## 功能说明

这是一个模块化的Kaggle数据采集工具，用于：
1. 根据关键词搜索Kaggle竞赛
2. 获取竞赛的排行榜数据
3. 下载排行榜用户的kernel代码

## 项目结构

```
.
├── config.json                  # 配置文件（关键词、参数等）
├── main.py                      # 主程序入口
├── utils.py                     # 工具函数（命令执行、CSV处理等）
├── search_competitions.py        # 竞赛搜索模块
├── get_leaderboard.py           # 排行榜获取模块
├── download_kernels.py          # kernel下载模块
└── README.md                    # 本文件
```

## 使用方法

### 1. 修改配置文件 (config.json)

```json
{
  "keywords": ["machine learning", "data science"],     # 默认关键词
  "output_dir": "output",                                # 输出目录
  "competitions_per_keyword": 5,                         # 每个关键词取5个竞赛
  "top_leaderboard_users": 5,                            # 排行榜取前5名
  "kernels_per_user": 5,                                 # 每个用户取5个kernel
  "save_csv": {
    "competitions": true,                                # 保存竞赛列表CSV
    "leaderboard": true,                                 # 保存排行榜CSV
    "kernels_list": true                                 # 保存kernel列表CSV
  },
  "sort_by_teams": "numberOfTeams"                       # 竞赛排序方式
}
```

**sort_by_teams 可选值**：
- `numberOfTeams`: 按参与人数排序（推荐，默认值）
- `prize`: 按奖金排序
- `earliestDeadline`: 按最早截止时间排序
- `latestDeadline`: 按最新截止时间排序
- `recentlyCreated`: 按最近创建排序
- `grouped`: 按分组排序

### 2. 运行程序

#### 方式A：使用配置文件中的关键词
```bash
python main.py
```

#### 方式B：使用命令行参数（优先级最高）
**关键词必须用引号包裹**，以支持多词关键词：
```bash
# 正确✅ - 单词关键词
python main.py "machine learning" "deep learning" "NLP"

# 正确✅ - 含有空格的关键词
python main.py "computer vision" "time series"

# 错误❌ - 不要这样写，会被分割成单个词
python main.py machine learning deep learning
```

## 流程说明

### 第一步：搜索竞赛
- 对每个关键词执行搜索
- 按参与人数从多到少排序
- 取前5个竞赛
- 保存竞赛列表为CSV（可选）

### 第二步：获取排行榜
- 对每个竞赛获取排行榜
- 取前5名用户
- 提取用户名（teamName或userName字段）
- 保存排行榜为CSV（可选）

### 第三步：下载Kernel
- 对每个用户查询其在该竞赛上的kernel
- 取前5个kernel（最新优先）
- 下载kernel代码和metadata
- 为每个kernel创建独立目录
- 保存kernel列表为CSV（可选）

## 输出目录结构

```
output/
├── <关键词>/
│   ├── <竞赛名>/
│   │   └── kernels/
│   │       ├── 1_kernel_name_1/
│   │       │   ├── kernel_file.ipynb
│   │       │   └── kernel-metadata.json
│   │       ├── 2_kernel_name_2/
│   │       └── ...
│   └── <竞赛名>/
└── csv_results/
    ├── competitions_*.csv
    ├── leaderboard_*.csv
    └── kernels_*.csv
```

## 模块说明

### utils.py
- `load_config()`: 加载JSON配置文件
- `run_kaggle_command()`: 执行kaggle CLI命令，支持自动重试
- `parse_csv_data()`: 解析CSV文本
- `save_csv()`: 保存数据为CSV文件
- `create_directory_structure()`: 创建输出目录结构
- `print_result()`: 打印数据结果

### search_competitions.py
- `search_competitions()`: 搜索单个关键词的竞赛
- `batch_search_competitions()`: 批量搜索多个关键词

### get_leaderboard.py
- `get_leaderboard()`: 获取竞赛排行榜
- `extract_usernames()`: 从排行榜提取用户名列表

### download_kernels.py
- `get_user_kernels()`: 获取用户在竞赛上的kernel列表
- `download_kernel()`: 下载单个kernel
- `download_competition_kernels()`: 下载竞赛所有用户的kernel

### main.py
- 主程序入口，按流程调用各个模块
- 处理命令行参数
- 输出统计结果

## 错误处理

### 网络问题
如果执行kaggle指令时遇到网络问题（超时、连接错误等），系统会自动重试3次，不需要修改代码。

### 日志输出
每一步都有详细的输出，包括：
- ✅ 成功操作
- ❌ 失败操作
- ⚠️ 警告信息
- 📊 统计信息
- 📁 文件操作

## 重要配置说明

| 参数 | 含义 | 可选值 | 默认值 |
|------|------|--------|--------|
| keywords | 搜索关键词列表 | 任意字符串 | ["machine learning", "data science"] |
| output_dir | 输出目录 | 任意路径 | "output" |
| competitions_per_keyword | 每个关键词取多少个竞赛 | 数字 | 5 |
| top_leaderboard_users | 排行榜取前多少名 | 数字 | 5 |
| kernels_per_user | 每个用户下载多少个kernel | 数字 | 5 |
| min_leaderboard_score | 排行榜最小分数阈值，低于该分数的用户将被过滤 | 小数（0-1）或 null（不启用） | 0.6 |
| sort_by_teams | 竞赛排序方式 | numberOfTeams, prize, earliestDeadline, latestDeadline, recentlyCreated, grouped | "numberOfTeams" |
| save_csv.competitions | 是否保存竞赛列表 | true/false | true |
| save_csv.leaderboard | 是否保存排行榜 | true/false | true |
| save_csv.kernels_list | 是否保存kernel列表 | true/false | true |

## 注意事项

1. **Kaggle API认证**：请确保已设置 `KAGGLE_API_TOKEN` 环境变量
2. **Conda环境**：请在运行程序前已激活所需的conda环境（如cameltrack）
3. **命令行参数**：所有关键词必须用**引号包裹**，例如 `python main.py "machine learning" "deep learning"`
4. **网络连接**：需要稳定的网络连接，建议在后台运行或使用tmux/screen
5. **磁盘空间**：大量kernel代码会占用较多磁盘空间，请确保有足够空间
6. **API速率限制**：Kaggle API可能有速率限制，程序会自动重试
7. **竞赛Slug格式**：程序会自动从竞赛ref中提取slug（如 `spaceship-titanic`），无需手动处理

## 示例运行

```bash
# 例1：单个关键词（记得用引号！）
python main.py "Titanic"

# 例2：多个关键词
python main.py "machine learning" "deep learning" "NLP"

# 例3：含空格的关键词
python main.py "computer vision" "time series" "anomaly detection"

# 例4：使用配置文件（无需引号）
python main.py
```
