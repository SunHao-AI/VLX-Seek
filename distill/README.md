# VLX-Seek → YOLO-World 蒸馏

用 VLX-Seek（teacher，细粒度感知 VLM）生成伪标签，蒸馏训练官方 YOLO-World（student，开放词汇检测器）。

## 目录结构

```
distill/
├── extract_image_urls.py       # 步骤1：从 json 中抽取 imageUrl 索引
├── download_images.py          # 步骤2：按 URL 列表并发下载图片
├── generate_prompts.py         # 步骤3：从检测服务生成 VLX-Seek 类别 prompt
├── generate_pseudo_labels.py   # 步骤4：VLX-Seek → COCO 伪标签
├── convert_annotations.py      # 标注格式互转：COCO / YOLO / LabelMe（6 个方向）
├── finetune_yolo_world.py      # 步骤5：COCO → 官方 YOLO-World 微调
├── coco_utils.py               # 共享 COCO 工具（坐标转换/划分/转 YOLO txt）
└── examples/                   # 端到端示例数据
    ├── images/                 #   demo 图片（demo_image.jpg / demo_image2.jpg）
    └── pseudo_labels.json      #   示例 COCO 伪标签（orange / apple）
```

## 环境要求

- **步骤1/2（抽取索引、下载图片）**：仅需 `requests`（`pip install requests`），纯 CPU 即可。
- **步骤3（伪标签生成）**：需要 GPU + VLX-Seek 权重（`resources/VLX-Seek-1.5-10B`）与 WeDetect 权重（`resources/wedetect_base_uni.pth`，缺失时自动下载）。依赖见项目根 `requirements.txt`。
- **步骤4（微调）**：需要额外安装 `ultralytics`（`pip install ultralytics`），建议独立虚拟环境，避免与项目 `torch 2.10 / transformers 5.13` 冲突。训练需足够内存/GPU。

## 步骤1：抽取图片索引

从标注 json 中抽取以指定前缀开头的 `imageUrl`，写入 URL 列表文件（每行一个）。

```bash
python distill/extract_image_urls.py <目标文件夹> <输出文件> [--workers N] [--prefix PREFIX]
```

- 递归遍历目标文件夹下所有 json 文件，匹配 `imageUrl` 字段。
- 默认前缀 `http://fsimage.guihuao.com`，可用 `--prefix` 覆盖。

## 步骤2：下载图片

读取 URL 列表文件，并发下载图片到指定目录；下载成功的 URL 会从列表中移除（失败/跳过的保留，便于重试）。

```bash
python distill/download_images.py <url_file> <download_dir> [选项]
```

- 支持 `-n/--num`（只下载前 N 张）、`--workers`（并发数）、`--dedup-mode`（去重模式）、`--timeout`、`--retries`、`--no-skip-existing`。
- 已存在且非空的文件默认跳过，可断点续跑。

## 步骤3：生成 VLX-Seek 类别 prompt

从检测服务获取全部类别信息，生成 VLX-Seek 推理用的类别 prompt 映射，供步骤4 伪标签生成使用。

```bash
python distill/generate_prompts.py
```

- 请求 `GET /v2/detect/all_class`，解析出全部 中文类别 <=> 英文类别 映射。
- 输出 `distill/data/category_prompts.json`，结构为 `{all_prompt, categories, prompt_to_category}`：
  - `all_prompt`：用 VLX-Seek 检测模板把全部类别 prompt 拼接而成，可直接用于整图开放词汇检测。
  - `categories`：每个中文类别含 `en_label`（英文名）、`prompt`（推理文本，默认中文名）、`models`（所属任务列表）。
  - `prompt_to_category`：反向映射 `{推理 prompt: 真实中文类别名}`，供步骤4 把输出 COCO 的 `categories.name` 还原为真实中文类别名。
- `prompt` 可手动改成更精确的描述（如 `"卫星锅"` → `"接收电视信号的卫星天线"`）；再次运行会保留手动修改，仅更新 `en_label`/`models`。
- 可选参数：`--url`（接口地址）、`--output`（输出路径）、`--timeout`。

## 步骤4：生成伪标签

```bash
python distill/generate_pseudo_labels.py \
  --image-dir data/images \
  --categories "person; car; dog" \
  --output data/pseudo_labels.json \
  --model-path resources/VLX-Seek-1.5-10B \
  --device cuda \
  --resume
```

- 每张图先用 WeDetect 生成候选区域（proposals），再调 VLX-Seek 开放词汇检测，输出 `{label, bbox}` 写入 COCO。
- 支持 `--start/--end-index` 分片、`--min-area` 过滤小框、`--gpu-ids` 多卡并行。
- **建议始终加 `--resume`**：脚本每 10 张图落盘一次，Ctrl+C / 进程中断最多丢失最近 10 张；`--resume` 会读取已有输出文件（多卡为各 `.shard<i>.json`），按 `file_name` 跳过已处理图像，只跑剩余部分，实现断点续跑。不加 `--resume` 则从头运行并覆盖旧结果。
- 类别按 `--categories` 精确匹配 VLX-Seek 输出的 label（忽略大小写），不匹配的框会被丢弃。
- 默认读取 `distill/data/category_prompts.json` 的 `prompt_to_category`，把输出 COCO 的 `categories.name` 从推理 prompt 替换为真实中文类别名；可用 `--prompt-map` 指定其他映射文件，文件缺失或缺少该字段时保持 prompt 原样。

### 使用全部类别 + 提示词分批

检测全部类别（`distill/data/category_prompts.json` 的 `all_prompt` 字段）时，提示词过长会导致 VLX-Seek 效果下降。此时应从 `all_prompt` 中提取全部类别并配合 `--prompt-batch-size` 分批循环推理：

```powershell
# 从 all_prompt 提取全部类别（去掉前缀 "Detect all the instances of: "，按分号分割）
# 输出分号分隔的类别列表，直接作为 --categories 传参
python -c "import json; data = json.load(open('distill/data/category_prompts.json', encoding='utf-8')); prompt = data['all_prompt']; body = prompt.split(': ', 1)[1].rstrip('.'); cats = [c.strip() for c in body.split(';') if c.strip()]; print('; '.join(cats))"
```

```bash
# 例如：280 个类别（当前 all_prompt 实际数量），每 30 个一组循环推理
# 默认启用滑窗裁剪推理：1000x1000 裁剪块 + 10% 重叠
uv run python distill/generate_pseudo_labels.py \
  --image-dir distill/data/images \
  --categories "人群密集; 道路施工区域或施工场景; 水面浑浊、不洁的水体; 水面漂浮的垃圾或废弃物; 黑色或散发臭味的水体; 在水中游泳的人; 靠近水域岸边的人员; 涉水行走或作业的人员; 消防通道堵塞; 河湖岸边搭建的房屋或构筑物; 农田中的建筑或构筑物; 搭建的充气拱门或气模装饰; 道路上的积水区域; 沿街晾晒的衣物或物品; 毁坏林地修建的坟墓; 堆放的废旧轮胎; 流动摊贩或路边小摊; 颜色异常的站立树木（枯死或病害）; 房屋破损、损坏的建筑; 破损的农业大棚或温室; 悬挂的气球或气球装饰; 烟囱或排放口的烟雾; 模糊不清的交通标志标线; 倒伏的庄稼或农作物; 秸秆焚烧后留下的痕迹; 屋顶积存的雨水或积水; 未覆盖篷布的渣土运输车; 水面生长的绿色藻类; 高速公路旁的广告牌; 接收卫星信号的圆形天线锅; 暴露在外的垃圾堆; 屋顶种植的绿色植物; 倒塌的房屋墙壁; 铁皮搭建的简易棚屋; 堆放的渣土或建筑余泥; 河道中采砂作业的船只; 道路破损、坑洼或裂缝; 正在施工的建设工程; 正在建造的农村自建房; 附属房屋的建筑材料堆放; 积存的建筑垃圾或废弃物; 道路上堆积的沙土; 荒草焚烧后留下的痕迹; 店铺门外占道经营; 山体滑坡风险区域; 悬挂的横幅或条幅; 工地中裸露的土壤; 屋顶上的垃圾或废弃物; 河道旁堆放的垃圾; 河道中的施工活动; 停放的机动车; 建筑工地上的塔式起重机; 路面施工设置的围挡; 水上航行的各类船舶; 道路防撞桶破损; 水面漂浮的油污; 森林中的大火或火灾; 监控视角下的道路积水; 海中生长的藻类植物; 建筑屋顶临时搭建或改造; 屋顶覆盖的黑色遮阳网; 消防栓渗水或漏水; 山林被砍伐后留下的痕迹; 现浇混凝土沟渠堵塞; 水位超过警戒线; 堤坝破损或损坏; 覆盖在裸土上的防尘网; 占用道路晒粮食; 建筑工地堆积的材料; 建筑外墙立面上的广告牌; 捕鱼作业的船只; 罐体运输车（油罐车、水泥罐车等）; 履带式或轮式挖掘机; 玻璃顶棚或阳光房; 装载机或铲车; 已覆盖篷布的渣土车; 汽车吊或履带吊等起重设备; 运输货物的船只; 公交车或大巴; 空载的渣土运输车; 货运卡车; 小轿车或家用汽车; 压路机或碾压设备; 叉车或铲车; 蓝色铁皮或彩钢瓦棚屋; 太阳能光伏板; 屋顶安装的太阳能热水器; 风力发电机故障或异常; 佩戴安全帽的人; 未佩戴安全帽的人; 玻璃材质的电力绝缘子; 红外图像中的光伏板缺陷; 红外图像中的绝缘套管; 红外图像中的绝缘子; 红外图像中的变压器; 电力线路上缺失的绝缘子; 违规放牧的羊群; 脏污的玻璃绝缘子; 缺失的玻璃绝缘子; 复合材料的电力绝缘子; 脏污的复合绝缘子; 成对的玻璃绝缘子; 绝缘子盘片破损或损坏; 电力线路上的绝缘子瓷瓶; 绝缘子表面污秽导致的闪络放电; 牛、羊等牲畜; 无人机编队表演; 燃放烟花爆竹的场景; 电力铁塔上的鸟窝; 设备或线路过热（红外热成像）; 电力变压器设备; 变压器连接的电力线路; 空的停车位; 被车辆占用的停车位; 被非车辆物体占用的停车位; 被非机动车占用的停车位; 乱停乱放占用停车位; 光伏板上的热斑异常; 停放的非机动车; 人行道区域; 屋顶安装的广告字; 店铺招牌或门头; 建筑立面安装的广告字; 屋顶安装的广告牌; 施工围挡上的广告牌; 封堵窗户的广告牌; 独立设置的户外广告牌; 建筑立面的电子显示媒体墙; 遮阳伞或户外伞棚; 店铺门口的遮阳棚; 佩戴安全帽的施工人员; 蓝色安全帽; 黄色安全帽; 红色安全帽; 白色安全帽; 施工产生的火花或飞溅; 反光背心或反光衣; 临边安全防护栏; 建筑工地中的基坑; 施工用梯笼; 防风加固措施; 水塘或水池; 工地活动板房; 施工隔离围挡; 红白相间的塑料水马隔离墩; 打桩机或桩工设备; 未佩戴安全帽的施工人员; 工业用气瓶或氧气瓶; 交通路锥或锥形桶; 高架桥的预制梁; 钢筋网片; 登高作业施工区域; 正在建设中的高架桥面; 进行电焊作业的人; 五点式安全带; 隔离围挡的间隔缺失; 倒伏的临边防护栏; 间隔缺失的临边防护栏; 缺少围挡的临边防护栏; 被堆放的临边防护栏; 竖向设置的临边防护栏; 建筑脚手架; 桥梁施工中的湿接缝; 湿接缝的盖板; 湿接缝的孔口; 横向设置的临边防护栏; 倒伏的梯笼; 水马或防撞桶; 倒伏的水马隔离墩; 禁止停车区域; 防撞架或防撞设施; 伸缩式护栏; 石墩或隔离墩; 未升起的阻车柱; 阻车柱或升降柱; 电动伸缩门; 减速带; 花箱或绿化箱; 学校门口的安防隔离带; 罂粟植株或花朵; 行人、站立的人或坐着的人; 水体水域区域; 水上浮式养殖网箱; 竹筏或竹排; 水上浮标; 水上起重船或浮吊; 码头上的起重机; 灯塔或航标灯; 客运船只; 巡逻船或执法船; 龙门架或门式起重机; 废弃的船只; 小型船只; 拖船或推船; 木质船只; 小型游艇或快艇; 大面积道路裂缝; 模糊的交通标注线; 块状脱落的外墙; 斑状脱落的外墙; 外墙渗水导致的霉斑; 消防栓或消火栓; 井盖或窨井盖; 山地梯田中的裸土区域; 山地梯田中的种植区域; 山地中的裸地; 合规停放的共享单车; 正在骑行中的共享单车; 在道路（不含斑马线）违停的共享单车; 在消防通道违停的共享单车; 在绿化带中违停的共享单车; 在斑马线上违停的共享单车; 多辆合规停放的共享单车; 多辆在道路（不含斑马线）违停的共享单车; 多辆在消防通道违停的共享单车; 多辆在绿化带违停的共享单车; 多辆在斑马线上违停的共享单车; 大量淤积在道路违停的共享单车集群; 大量在消防通道违停的共享单车集群; 大量在绿化带违停的共享单车集群; 大量在斑马线上违停的共享单车集群; 大量合规停放的共享单车集群; 大量呈线状在道路违停的共享单车集群; 在人行道上停放的共享单车; 多辆在人行道上停放的共享单车; 大量在人行道上停放的共享单车集群; 大量散乱停放在人行道上的共享单车集群; 黑色遮阳网或防护网; 夜间场景中的小客车; 夜间场景中的货船; 夜间场景中的混凝土搅拌车; 夜间场景中的挖掘机; 夜间场景中的摩托车; 夜间场景中的人; 夜间场景中的货车; 电瓶车或电动自行车; 加装车篷的电瓶车; 加装车篷的电动三轮车; 骑电瓶车的人; 电动三轮车; 骑电瓶车违规载人; 堆放的电瓶车; 骑电瓶车已佩戴头盔; 骑电瓶车未佩戴头盔; 共享电瓶车或共享单车; 堆放的共享电瓶车; 三轮车（人力或电动）; 山体开挖区域; 建筑工地的大门; 工地出入口的过水槽; 农用机械或拖拉机; 大麻植株; 水泥地面的横向裂缝; 水泥地面的网状裂缝; 水泥地面的纵向裂缝; 沥青路面的横向裂缝; 沥青路面的网状裂缝; 路面条状修补痕迹; 交通标线被污染; 交通标线磨损; 沥青路面的纵向裂缝; 狗或犬只; 警车; 救护车; 消防车; 安全防护网; 破损的安全防护网; 裸露的土壤或地面; 脏乱的绿地或草坪; 登高作业时未佩戴安全帽; 学校门口区域; 挖掘机近场区域有人员侵入; 吊车近场区域有人员侵入; 进行登高作业的工人; 坑洞旁的临边安全防护; 经过加固的梯笼; 基坑中的积水; 登高作业时未佩戴安全绳; 未穿反光背心的人员; 穿着反光衣的人员" \
  --output distill/data/pseudo_labels.json \
  --model-path resources/VLX-Seek-1.5-10B \
  --prompt-batch-size 50 \
  --gpu-ids "0,1,2,3,4,5,6,7" \
  --slice-width 2500 --slice-height 2500 \
  --queue-batch-size 1 \
  --resume
```

- `--prompt-batch-size N`：每个子提示词最多包含 N 个类别，类别超过 N 时自动拆成多个子提示词循环推理（如 280 个类别 + `--prompt-batch-size 30` → 10 轮）。
- 分批模式下同一张图（或裁剪块）的图片特征只编码一次（视觉编码器 + 文本 embedding 复用），仅文本提示词随批次变化，可显著减少重复视觉编码开销。
- 默认值为 0，表示不拆分，行为与未启用该参数一致。
- 注意：`--categories` 传的是类别名列表（分号分隔），不是整段 `all_prompt` 文本；`all_prompt` 开头的 `Detect all the instances of:` 前缀由脚本内部的检测模板自动添加。

### 多卡并行（指定部分 GPU）

`--gpu-ids` 支持任意的 GPU 索引组合（不要求连续），每个索引启动一个子进程并加载一个独立模型：

```bash
# 8 卡机器上只用物理 GPU 0/5/6/7，启动 4 个模型并行推理
python distill/generate_pseudo_labels.py \
  --image-dir data/images \
  --categories "人群密集; 道路施工区域或施工场景; 水面浑浊、不洁的水体; ..." \
  --output data/pseudo_labels.json \
  --model-path resources/VLX-Seek-1.5-10B \
  --prompt-batch-size 30 \
  --gpu-ids "0,5,6,7" \
  --resume
```

- 每个子进程通过 `CUDA_VISIBLE_DEVICES=<gpu_id>` 隔离到对应物理 GPU，再以 `cuda:0` 使用它。
- 图像按轮询方式均分到各 GPU；各卡结果写入 `<output>.shard<i>.json`，全部完成后合并为最终输出（分片文件保留，可配合 `--resume` 断点续跑）。
- 多卡模式与 `--prompt-batch-size` 兼容：每个子进程内部同样走分批 + 图片编码复用逻辑。
- 注意显存：每个进程需加载完整 VLX-Seek 模型（10B）+ WeDetect，N 个进程同时驻留需约 N × 20GB+ 显存。

## 标注格式转换（convert_annotations.py）

步骤4 输出的伪标签是 COCO 格式；微调前可能需要在 COCO / YOLO / LabelMe 之间互转（如转换后交给标注平台人工修正、或导出 YOLO txt 直接训练）。本脚本以统一中间表示实现 6 个转换方向：

```
COCO   ⇄  YOLO（检测/分割 txt）
COCO   ⇄  LabelMe
YOLO   ⇄  LabelMe
```

```bash
python distill/convert_annotations.py <子命令> [参数]
python distill/convert_annotations.py --help        # 查看全部子命令
python distill/convert_annotations.py <子命令> --help   # 查看子命令参数
```

依赖仅标准库 + PIL（图片尺寸读取），无需额外安装。

### COCO → YOLO

```bash
python distill/convert_annotations.py coco2yolo \
  --coco-json distill/data/pseudo_labels.json \
  --image-dir distill/data/images \
  --out-dir distill/data/yolo
```

- `--image-dir`：COCO `file_name` 相对此目录的图片源目录，用于复制图片与读取尺寸。
- 输出目录结构（ultralytics 兼容）：

```
<out-dir>/
├── images/          # 复制的图片（加 --no-copy-images 则不复制）
├── labels/          # 每图一个 txt：class_id cx cy w h（检测）或 + 多边形点（分割，归一化 6 位小数）
├── names.txt        # 每行一个类别名，顺序即 class_id
└── dataset.yaml     # 可直接用于 YOLO 训练
```

### YOLO → COCO

```bash
python distill/convert_annotations.py yolo2coco \
  --image-dir distill/data/yolo/images \
  --label-dir distill/data/yolo/labels \
  --names distill/data/yolo/names.txt \
  --out-json distill/data/pseudo_labels_coco.json
```

### COCO → LabelMe

```bash
python distill/convert_annotations.py coco2labelme \
  --coco-json distill/data/pseudo_labels.json \
  --out-dir distill/data/labelme
```

- 每张图输出一个同名 `.json`（bbox 转为 `rectangle`，segmentation 转为 `polygon`），`imageData` 为 `null`（不含内嵌图片），可配合 `labelme` 工具人工修正。

### LabelMe → COCO

```bash
python distill/convert_annotations.py labelme2coco \
  --labelme-dir distill/data/labelme \
  --out-json distill/data/pseudo_labels_coco.json
```

### YOLO → LabelMe

```bash
python distill/convert_annotations.py yolo2labelme \
  --image-dir distill/data/yolo/images \
  --label-dir distill/data/yolo/labels \
  --names distill/data/yolo/names.txt \
  --out-dir distill/data/labelme
```

### LabelMe → YOLO

```bash
python distill/convert_annotations.py labelme2yolo \
  --labelme-dir distill/data/labelme \
  --image-dir distill/data/images \
  --out-dir distill/data/yolo
```

- `--image-dir` 缺省时相对当前目录查找图片；加 `--no-copy-images` 则不复制图片到 `out/images`。

### 通用说明

- **`--names` 支持两种格式**：`names.txt`（每行一个类别名）或 `data.yaml`（含 `names:` 键，兼容多行列表与内联 `{0: name, 1: name}` 写法）。
- **类别顺序**：转出方向（→ YOLO / → LabelMe）按类别首次出现顺序自动分配 class_id，并写出 `names.txt` 固化顺序；转回方向需用同一份 `names.txt` 保证 id 一致。
- **边界数据自动跳过并告警**（不中断转换）：非法/越界 bbox、RLE 分割、空标注文件、缺 `imageWidth`/`imageHeight` 的 LabelMe、无对应图片的标注等。运行结束后汇总输出警告条数，可据此检查源数据。

## 步骤5：微调 YOLO-World

```bash
python distill/finetune_yolo_world.py \
  --coco-json data/pseudo_labels.json \
  --image-dir data/images \
  --weights yolov8s-worldv2.pt \
  --epochs 50 --imgsz 640 --batch 16 --device 0
```

- 自动划分 train/val（或传 `--val-coco-json` 指定独立验证集），转成 YOLO txt + 生成 `dataset.yaml`，调用 `YOLOWorld.train()`。
- 首次运行会下载 YOLO-World 预训练权重与 CLIP 文本编码器权重。

## 端到端示例（examples/）

`examples/` 提供最小可运行示例：2 张 demo 图片 + 一份示例 COCO 伪标签（`orange` / `apple`）。

```bash
# 用示例伪标签微调（CPU 小规模验证流程）
python distill/finetune_yolo_world.py \
  --coco-json distill/examples/pseudo_labels.json \
  --image-dir distill/examples/images \
  --output-dir distill/examples/runs \
  --weights yolov8s-worldv2.pt \
  --epochs 1 --imgsz 320 --batch 1 --device cpu --workers 0 --val-ratio 0.5
```

> 说明：`examples/pseudo_labels.json` 是手工构造的示例，用于验证微调流程；真实场景请用步骤1 由 VLX-Seek 生成。

## 注意事项

1. **伪标签质量受 proposal 召回限制**：VLX-Seek 是"区域检索"而非"坐标回归"，只能从 WeDetect 提出的候选框中选择。若 WeDetect 漏检目标，伪标签会系统性漏检。可提高 `num_proposals` 或换更强的 proposal 生成器。
2. **CLIP 权重下载**：微调需下载 CLIP 文本编码器权重（约 337MB）。若自动下载校验失败（网络截断），可手动下载到 ultralytics 的 `weights/clip/` 目录。
3. **内存/GPU**：YOLO-World 训练与 VLX-Seek 推理都需要较大内存/GPU，CPU 小内存环境可能无法完成完整训练。
