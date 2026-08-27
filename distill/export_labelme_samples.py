#!/usr/bin/env python
"""从 COCO 伪标签 + 清洗决策日志中抽取 N 张图，转换为 LabelMe 格式便于人工抽查标注质量。

LabelMe 的 shape.label 会拼上 verdict 后缀（如「小轿车或家用汽车|keep」），
在 LabelMe 中不同 label 会显示为不同颜色，一眼区分 keep / delete。

用法:
  uv run python distill/export_labelme_samples.py \
      --coco-json distill/data/pseudo_labels.json \
      --image-dir distill/data/images \
      --decisions distill/data/pseudo_labels.decisions.jsonl \
      --num 12 --seed 0 \
      --outdir runs/labelme_samples
"""
import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco-json', required=True)
    ap.add_argument('--image-dir', required=True)
    ap.add_argument('--decisions', required=True)
    ap.add_argument('--num', type=int, default=12, help='抽取图片张数')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--verdict', choices=['keep', 'delete', 'error_keep'],
                    help='可选：仅抽取含该 verdict 标注的图片（不填则任意）')
    args = ap.parse_args()

    coco = json.load(open(args.coco_json, encoding='utf-8'))
    images = {i['id']: i for i in coco['images']}
    cats = {c['id']: c['name'] for c in coco['categories']}
    by_img = defaultdict(list)
    an_to_ann = {}
    for a in coco['annotations']:
        by_img[a['image_id']].append(a)
        an_to_ann[a['id']] = a

    # ann_id -> verdict；同时统计每张图含有的 verdict 集合
    av = {}
    img_has = defaultdict(set)
    for line in open(args.decisions, encoding='utf-8'):
        r = json.loads(line)
        if 'verdict' not in r:
            continue
        ans = an_to_ann.get(r.get('ann_id'))
        if ans is None:
            continue
        img_has[ans['image_id']].add(r['verdict'])
        av[r['ann_id']] = r['verdict']

    # 候选图片
    cand = list(img_has.keys()) if args.verdict else list(by_img.keys())
    if args.verdict:
        cand = [i for i in cand if args.verdict in img_has[i]]
    random.seed(args.seed)
    random.shuffle(cand)
    sel = cand[:args.num]

    os.makedirs(args.outdir, exist_ok=True)
    used = 0
    for img_id in sel:
        im = images[img_id]
        fname = im['file_name']
        src = Path(args.image_dir) / fname
        if not src.exists():
            print('!! 图片缺失，跳过', fname)
            continue
        shutil.copy2(src, Path(args.outdir) / fname)

        shapes = []
        by_v = defaultdict(int)
        for a in by_img[img_id]:
            cat = cats.get(a['category_id'], 'unknown')
            v = av.get(a['id'], 'unknown')
            by_v[v] += 1
            x, y, w, h = a['bbox']
            shapes.append({
                'label': f'{cat}|{v}',
                'points': [[x, y], [x + w, y + h]],
                'group_id': None,
                'shape_type': 'rectangle',
                'flags': {},
            })
        labelme = {
            'version': '5.5.0',
            'flags': {},
            'shapes': shapes,
            'imagePath': fname,
            'imageData': None,
            'imageHeight': im.get('height'),
            'imageWidth': im.get('width'),
        }
        out_json = Path(args.outdir) / (Path(fname).stem + '.json')
        json.dump(labelme, open(out_json, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        keep_cnt = by_v.get('keep', 0)
        del_cnt = by_v.get('delete', 0)
        err_cnt = by_v.get('error_keep', 0)
        print(f'导出 {fname}: keep={keep_cnt} delete={del_cnt} error_keep={err_cnt} 标注x{len(shapes)}')
        used += 1

    print('---')
    print(f'共导出 {used} 张图到 {args.outdir}')
    print(f'其中标注总数: {sum(len(by_img[i]) for i in sel)}')


if __name__ == '__main__':
    main()
