#!/usr/bin/env python3
"""
极简 ACL 论文下载器
for actually_pipeline 教程

用法：改下面 CONFIG，然后 python src/download.py
"""

import asyncio
import json
import random
import re
import time
from pathlib import Path

import aiohttp
import requests
from tqdm.asyncio import tqdm


# ==================== CONFIG（只改这里） ====================
# 每个 volume 的网页前缀 -> 本地缓存文件夹名
# 已有的缓存：acl2026_long, acl2026_short, acl2026_findings
VOLUMES = [
    {"prefix": "2026.acl-long",   "cache": "acl2026_long"},
    {"prefix": "2026.acl-short",  "cache": "acl2026_short"},
    {"prefix": "2026.findings-acl", "cache": "acl2026_findings"},
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / "data" / ".cache"

CONCURRENCY = 2
DELAY = (0.5, 1.5)
# ==========================================================


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_papers(prefix: str):
    url = f"https://aclanthology.org/volumes/{prefix}/"
    log(f"获取: {prefix}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    html = resp.text

    pattern = re.compile(r'\b' + re.escape(prefix) + r'\.(\d+)\b', re.IGNORECASE)
    nums = sorted(set(int(n) for n in pattern.findall(html)))

    titles = {}
    title_re = re.compile(
        r'href=/?' + re.escape(prefix) + r'\.(\d+)/?>([^<]+)</a>',
        re.IGNORECASE,
    )
    for m in title_re.findall(html):
        titles[int(m[0])] = m[1].strip()

    papers = []
    for n in nums:
        pid = f"{prefix}.{n}"
        papers.append({
            "id": pid,
            "title": titles.get(n, ""),
            "pdf_url": f"https://aclanthology.org/{pid}.pdf",
            "sort_key": n,
        })
    return papers


async def download_one(session, paper, cache_dir, pbar, stats):
    pid = paper["id"]
    url = paper["pdf_url"]
    dest = cache_dir / (pid.replace("/", "_") + ".pdf")

    if dest.exists() and dest.stat().st_size > 1024:
        pbar.update(1)
        stats["skip"] += 1
        return {"id": pid, "status": "ok", "path": str(dest)}

    for attempt in range(1, 4):
        try:
            timeout = aiohttp.ClientTimeout(total=60, connect=5, sock_read=30)
            async with session.get(url, timeout=timeout) as resp:
                if resp.status == 200:
                    dest.write_bytes(await resp.read())
                    pbar.update(1)
                    stats["ok"] += 1
                    return {"id": pid, "status": "ok", "path": str(dest)}
        except Exception:
            pass
        await asyncio.sleep(2 * attempt)

    pbar.update(1)
    stats["fail"] += 1
    log(f"  [FAIL] {pid}")
    return {"id": pid, "status": "fail", "url": url}


async def download_volume(prefix: str, cache_name: str):
    cache_dir = CACHE_ROOT / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    papers = fetch_papers(prefix)
    if not papers:
        log(f"[!] {prefix}: 没论文")
        return

    meta = cache_dir / "metadata.json"
    meta.write_text(json.dumps(papers, indent=2, ensure_ascii=False), encoding="utf-8")

    connector = aiohttp.TCPConnector(limit=4, limit_per_host=1)
    async with aiohttp.ClientSession(connector=connector) as session:
        pbar = tqdm(total=len(papers), desc=f"下载 {cache_name}", unit="篇")
        sem = asyncio.Semaphore(CONCURRENCY)
        stats = {"ok": 0, "skip": 0, "fail": 0}

        async def wrap(p):
            async with sem:
                await asyncio.sleep(random.uniform(*DELAY))
                return await download_one(session, p, cache_dir, pbar, stats)

        results = await asyncio.gather(*[wrap(p) for p in papers])
        pbar.close()

    # 二次重试
    failed = [p for p, r in zip(papers, results) if r["status"] == "fail"]
    if failed:
        log(f"  重试 {len(failed)} 篇...")
        connector = aiohttp.TCPConnector(limit=2, limit_per_host=1)
        async with aiohttp.ClientSession(connector=connector) as session:
            pbar = tqdm(total=len(failed), desc=f"重试 {cache_name}", unit="篇")
            sem = asyncio.Semaphore(1)
            stats2 = {"ok": 0, "skip": 0, "fail": 0}

            async def wrap2(p):
                async with sem:
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                    return await download_one(session, p, cache_dir, pbar, stats2)

            retry_results = await asyncio.gather(*[wrap2(p) for p in failed])
            pbar.close()

        ok_map = {r["id"]: r for r in results if r["status"] == "ok"}
        for r in retry_results:
            if r["status"] == "ok":
                ok_map[r["id"]] = r
        results = list(ok_map.values()) + [r for r in results if r["status"] != "ok" and r["id"] not in ok_map]

    ok = sum(1 for r in results if r["status"] == "ok")
    fail = sum(1 for r in results if r["status"] == "fail")
    log(f"  {cache_name}: 成功={ok}, 失败={fail}")

    if fail:
        (cache_dir / "failed.json").write_text(
            json.dumps([r for r in results if r["status"] == "fail"], indent=2),
            encoding="utf-8",
        )


async def main():
    log("=" * 40)
    log(f"准备下载 {len(VOLUMES)} 个 volume")
    log(f"缓存根目录: {CACHE_ROOT}")
    log("=" * 40)

    for cfg in VOLUMES:
        await download_volume(cfg["prefix"], cfg["cache"])

    log("[✓] 全部完成")
    log("下一步: python src/extract_text.py")


if __name__ == "__main__":
    asyncio.run(main())