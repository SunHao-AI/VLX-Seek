#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遍历目标文件夹(含子目录)下的所有 json 文件,
找出 imageUrl 字段以指定前缀开头的记录, 并将该字段值写入输出文件(每行一个).

用法:
    python extract_image_urls.py <目标文件夹> <输出文件> [--workers N] [--prefix PREFIX]

示例:
    python extract_image_urls.py ./labels ./urls.txt --workers 16
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 注意: 实际数据中反引号包裹的是整个 URL, 如
#   @url:`https://fsimage.guihuao.com/images/xxx.jpg`
# 反引号在 .com 之后并不存在(在文件名之后才闭合), 因此默认前缀不带末尾反引号。
# 若确实需要按 "com 后带反引号" 的写法匹配, 可传 --prefix "@url:`https://fsimage.guihuao.com`"
DEFAULT_PREFIX = "http://fsimage.guihuao.com"


def find_json_files(root: str):
    """递归遍历 root, 产出所有 .json 文件的路径."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".json"):
                yield os.path.join(dirpath, name)


def extract_image_url(file_path: str, prefix: str):
    """读取单个 json 文件, 若 imageUrl 以 prefix 开头则返回该值, 否则返回 None."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        url = data.get("imageUrl")
        if isinstance(url, str) and url.startswith(prefix):
            return url
    except Exception as e:
        print(f"[跳过] {file_path}: {e}", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="提取 json 文件中以指定前缀开头的 imageUrl 字段")
    parser.add_argument("folder", help="目标文件夹(递归遍历其中的 json 文件)")
    parser.add_argument("output", help="输出文件路径")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                        help="imageUrl 前缀过滤条件(默认: %(default)s)")
    parser.add_argument("--workers", type=int, default=8,
                        help="线程数(默认 8)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"错误: 文件夹不存在: {args.folder}", file=sys.stderr)
        sys.exit(1)

    files = list(find_json_files(args.folder))
    if not files:
        print("未找到任何 json 文件.")
        return
    print(f"共找到 {len(files)} 个 json 文件, 使用 {args.workers} 个线程处理...")

    write_lock = threading.Lock()
    found = 0
    done = 0

    with open(args.output, "w", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:

        futures = {pool.submit(extract_image_url, f, args.prefix): f
                   for f in files}
        for future in as_completed(futures):
            done += 1
            url = future.result()
            if url:
                with write_lock:
                    out.write(url + "\n")
                    found += 1
            if done % 1000 == 0:
                print(f"进度: {done}/{len(files)} 文件, 已匹配 {found} 条")

    print(f"完成! 共处理 {len(files)} 个 json 文件, "
          f"匹配 {found} 条, 已写入: {args.output}")


if __name__ == "__main__":
    main()
