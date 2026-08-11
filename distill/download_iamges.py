# -*- coding: utf-8 -*-
"""
图片下载函数模块。

核心函数 download_images(url_file, download_dir) 会：
  1. 读取 url_file 中的每一行 URL
  2. 先去重（默认精确去重，可改用大小写不敏感或按图片ID去重）
  3. 并发下载图片到 download_dir（失败自动重试、已存在的文件自动跳过）
  4. 下载成功的 URL 会从 url_file 中移除（失败/跳过的保留，便于下次重试）

用法（手动触发下载）：
    >>> import download_images as di
    >>> di.download_images("urls.txt", "D:/我的图片")           # 最简单的调用
    >>> di.download_images("urls.txt", "images", workers=16)    # 指定并发数
    >>> di.download_images("urls.txt", "images", n=100)         # 只下载前100张

    或者在文件末尾的 if __name__ == "__main__" 里填好路径后直接运行本文件。
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    raise ImportError("缺少依赖 requests，请先执行: pip install requests")

# 匹配 URL 末尾的数字图片ID，形如 .../images/<uuid>/5107861144794566657.jpeg
_ID_RE = re.compile(r"(?:^|/)([0-9]+)\.[a-zA-Z0-9]+$")

# 常见图片扩展名，用于生成本地文件名
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif", ".svg", ".heic", ".heif"}


def read_urls(url_file):
    """读取 url_file，返回去除空白后的 URL 列表（保留原始顺序）。"""
    urls = []
    with open(url_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip(" \t\r\n")
            if line:
                urls.append(line)
    return urls


def _dedup_key(url, mode):
    """根据去重模式生成 key。"""
    if mode == "exact":
        return url
    if mode == "casefold":
        return url.casefold()
    if mode == "id":
        m = _ID_RE.search(url)
        return m.group(1) if m else url.casefold()
    raise ValueError(f"未知去重模式: {mode}")


def dedup(urls, mode="exact"):
    """
    去重并保留首次出现顺序。
    返回: (去重后列表, 被移除的重复数量)
    mode: 'exact' 精确 / 'casefold' 忽略大小写 / 'id' 按末尾图片ID
    """
    seen, result, removed = set(), [], 0
    for u in urls:
        key = _dedup_key(u, mode)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        result.append(u)
    return result, removed


def _safe_name(index, url):
    """由序号 + URL 生成本地文件名，避免同名冲突。"""
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if ext not in _IMAGE_EXTS:
        ext = ".img"
    return f"{index:06d}{ext}"


def _download_one(task):
    """下载单个 URL，返回 (url, ok, reason)。"""
    url, dest_dir, timeout, retries, index, skip_existing = task
    dest = os.path.join(dest_dir, _safe_name(index, url))

    if skip_existing and os.path.exists(dest) and os.path.getsize(dest) > 0:
        return url, True, "already-exists"

    headers = {"User-Agent": "Mozilla/5.0 (compatible; ImageDownloader/1.0)"}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
                if r.status_code != 200:
                    return url, False, f"HTTP {r.status_code}"
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, dest)  # 原子替换，避免半截文件
                return url, True, "ok"
        except Exception as e:
            last_err = e
            time.sleep(min(0.5 * attempt, 3))
    return url, False, f"{type(last_err).__name__}: {last_err}"


def _remove_done(url_file, done_urls):
    """
    把已成功下载的 URL 从 url_file 中移除，保留其余行（顺序不变）。
    done_urls 为去重后的 URL（已 strip），与文件中的行按 strip 后内容匹配。
    返回被移除的数量。
    """
    if not done_urls:
        return 0
    done = set(done_urls)
    kept = []
    removed = 0
    with open(url_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip(" \t\r\n")
            if stripped and stripped in done:
                removed += 1  # 该行已下载完成，移除
            else:
                kept.append(line.rstrip("\n"))
    # 写回文件（去掉末尾多余空行）
    with open(url_file, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n" if kept else "")
    return removed


def download_images(url_file, download_dir, n=None, workers=8,
                    dedup_mode="exact", timeout=15, retries=3, skip_existing=True):
    """
    手动触发下载：读取 url_file，去重后并发下载图片到 download_dir，
    下载成功的 URL 会从 url_file 中移除。

    参数:
        url_file      : 包含图片 URL 的文本文件路径
        download_dir  : 图片保存目录（不存在会自动创建）
        n             : 本次只下载前 N 张（去重后）。None 表示全部
        workers       : 并发下载线程数，默认 8
        dedup_mode    : 'exact' 精确 / 'casefold' 忽略大小写 / 'id' 按图片ID
        timeout       : 单次请求超时秒数，默认 15
        retries       : 失败重试次数，默认 3
        skip_existing : 跳过已存在且非空的文件，默认 True

    返回:
        dict: {"total": 去重后总数, "download": 本次尝试下载数,
               "ok": 成功数, "fail": 失败数, "skipped": 已存在跳过数,
               "removed": 从文件移除的URL数, "failed_urls": 失败URL列表}
    """
    # 1) 读取 + 去重
    urls = read_urls(url_file)
    deduped, removed = dedup(urls, dedup_mode)
    print(f"读取 {len(urls):,} 个 URL，去重后 {len(deduped):,} 个（移除重复 {removed:,} 个）")

    # 2) 截取前 N 张
    to_download = deduped if n is None else deduped[:n]
    print(f"本次下载 {len(to_download):,} 张" + (f"（前 {n} 张）" if n is not None else "（全部）"))

    # 3) 下载
    os.makedirs(download_dir, exist_ok=True)
    tasks = [(u, download_dir, timeout, retries, i, skip_existing)
             for i, u in enumerate(to_download, 1)]

    ok = fail = skipped = 0
    done_urls, failed_urls = [], []
    t0 = time.time()
    print(f"开始下载 {len(tasks):,} 张图片 -> {download_dir} (并发 {workers}) ...")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_download_one, t) for t in tasks]
        for idx, fut in enumerate(as_completed(futures), 1):
            url, success, reason = fut.result()
            if reason == "already-exists":
                skipped += 1
            elif success:
                ok += 1
                done_urls.append(url)
            else:
                fail += 1
                failed_urls.append(url)
            if idx % 1000 == 0:
                print(f"  进度 {idx:,}/{len(tasks):,} (成功 {ok:,} 失败 {fail:,} 跳过 {skipped:,})")

    # 4) 从 url_file 中移除已成功下载的 URL（失败/跳过的保留，便于重试）
    removed_from_file = _remove_done(url_file, done_urls)
    print("\n下载完成："
          f"成功 {ok:,} | 失败 {fail:,} | 已存在跳过 {skipped:,} | 耗时 {time.time()-t0:.1f} 秒")
    print(f"已从 {url_file} 移除 {removed_from_file:,} 条已下载 URL，剩余 {max(len(urls)-removed_from_file,0):,} 条")
    return {
        "total": len(deduped),
        "download": len(to_download),
        "ok": ok,
        "fail": fail,
        "skipped": skipped,
        "removed": removed_from_file,
        "failed_urls": failed_urls,
    }


if __name__ == "__main__":
    # ====== 在这里填好路径，然后运行本文件即可手动触发下载 ======
    INPUT_FILE = "urls.txt"          # 输入：URL 列表文件
    OUTPUT_DIR = "images"            # 下载：图片保存文件夹
    N = 100                          # 本次只下载前 N 张；None 表示全部
    result = download_images(INPUT_FILE, OUTPUT_DIR, n=N, workers=8)
    # 若需要查看失败项，可打印 result["failed_urls"]