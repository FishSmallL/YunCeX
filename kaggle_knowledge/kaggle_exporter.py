#!/usr/bin/env python3
"""
Kaggle competitions exporter (uses -p page, stable)
"""

import subprocess
import csv
import sys
from io import StringIO

# ========== 配置参数 ==========
CATEGORY = "all"  # all / featured / playground / research / recruitment
SORT_BY = "latestDeadline"  # prize / numberOfTeams / latestDeadline / earliestDeadline
TOTAL = 10  # 想要多少条
OUTPUT = "kaggle_competitions.csv"
# ========================================

def fetch_competitions():
    """获取Kaggle竞赛列表并导出CSV"""
    all_rows = []
    page = 1

    while len(all_rows) < TOTAL:
        print(f"📄 Fetching page {page} ...", end=" ", flush=True)

        # 调用 kaggle CLI
        cmd = [
            "kaggle", "competitions", "list",
            "--category", CATEGORY,
            "--sort-by", SORT_BY,
            "-p", str(page),
            "--csv"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            raw_data = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error running kaggle command: {e.stderr}")
            sys.exit(1)

        # 检查是否有数据
        if not raw_data or len(raw_data.splitlines()) <= 1:
            print("\n⚠️ No more pages.")
            break

        # 解析CSV
        try:
            csv_file = StringIO(raw_data)
            reader = csv.DictReader(csv_file)
            rows = list(reader)

            if not rows:
                break

            all_rows.extend(rows)
            print(f"✓ Done, total: {len(all_rows)}")
            page += 1

        except csv.Error as e:
            print(f"\n❌ CSV parsing error: {e}")
            break

    # 截取到所需数量
    all_rows = all_rows[:TOTAL]

    # 导出CSV
    if all_rows:
        keys = all_rows[0].keys()
        with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"\n✅ Finished! Exported {len(all_rows)} → {OUTPUT}")
    else:
        print("\n⚠️ No data exported.")

if __name__ == "__main__":
    fetch_competitions()
