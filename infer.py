#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer.py — RoboMaster 装甲板 YOLO26n-pose 推理脚本

模型 best.pt 的实际情况（已从文件内读出）：
    ultralytics   8.4.174
    task          pose（关键点任务）
    网络          YOLO26n-pose，由 yolo26n-pose.pt 蒸馏训练而来（teacher 是 yolo26x-pose）
    类别          6 类：B1 B2 B3 R1 R2 R3   (B=蓝方, R=红方；1/2/3=装甲板大小)
    关键点        kpt_shape = [4, 2]，每块装甲板 4 个角点
    输入          640x640，输出端已内置 NMS（end2end=True），iou 默认 0.7

用法：
    python infer.py                          # 推理 best.pt 同目录下的 test.jpg
    python infer.py -s armor.jpg             # 推理单张图片
    python infer.py -s images/               # 推理整个文件夹
    python infer.py -s demo.mp4              # 推理视频
    python infer.py -s 0                     # 调用摄像头
    python infer.py -s armor.jpg --conf 0.5  # 自定义置信度阈值
    python infer.py -s armor.jpg --no-show   # 不弹窗，只存文件

输出（都放在 runs/infer/ 下）：
    <同名>.jpg    画好检测框和 4 个角点的图片
    <同名>.txt    每个目标一行，便于程序读取：
                  类别 置信度 x1 y1 x2 y2 | kpt: ax,ay bx,by cx,cy dx,dy | kpt_conf: ...
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "best.pt"
DEFAULT_OUTPUT = HERE / "runs" / "infer"

# 类别中文名，只是打印时好看
CLASS_CN = {
    "B1": "蓝方1号装甲", "B2": "蓝方2号装甲", "B3": "蓝方3号装甲",
    "R1": "红方1号装甲", "R2": "红方2号装甲", "R3": "红方3号装甲",
}

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")


def info(msg):
    print(f"[infer] {msg}", flush=True)


def die(msg):
    print(f"\n[错误] {msg}\n", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO26n-pose 装甲板推理",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", default=str(DEFAULT_MODEL),
                   help="模型路径，默认脚本同目录的 best.pt")
    p.add_argument("-s", "--source", default=None,
                   help="图片/文件夹/视频/摄像头(0)，默认 best.pt 同目录的 test.jpg")
    p.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT),
                   help="结果保存目录，默认 ./runs/infer")
    p.add_argument("--conf", type=float, default=0.35,
                   help="置信度阈值，默认 0.35（本模型自带的 conf 为空，这里给个稳妥值）")
    p.add_argument("--iou", type=float, default=0.70,
                   help="NMS 的 IoU 阈值，默认 0.70（与模型训练设置一致）")
    p.add_argument("--imgsz", type=int, default=640,
                   help="推理分辨率，默认 640（与训练一致）")
    p.add_argument("--device", default=None,
                   help="cpu 或 0(第一块显卡)；默认自动选择")
    p.add_argument("--max-det", type=int, default=300,
                   help="最多检测多少个目标，默认 300")
    p.add_argument("--no-show", action="store_true",
                   help="不弹出预览窗口")
    p.add_argument("--no-txt", action="store_true",
                   help="不保存坐标 txt 文件")
    p.add_argument("--center", action="store_true",
                   help="额外输出每块装甲板 4 个角点的中心坐标（做自瞄时有用）")
    return p.parse_args()


def load_model(model_path):
    if not Path(model_path).exists():
        die(f"找不到模型文件: {model_path}")

    try:
        from ultralytics import YOLO
    except ImportError:
        die("没有安装 ultralytics。请先执行:\n"
            "    pip install -U ultralytics\n"
            "本模型由 ultralytics 8.4.174 训练，版本不能低于它。")

    info(f"加载模型: {model_path}")
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        msg = str(e)
        if "ultralytics" in msg.lower() and "version" in msg.lower():
            die("模型由较新的 ultralytics 保存，当前版本太低。请执行:\n"
                "    pip install -U 'ultralytics>=8.4.174'\n"
                f"原始报错: {msg}")
        die(f"模型加载失败: {msg}")

    # 打印模型自带信息，确认加载的是预期模型
    try:
        names = model.names
        labels = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        info(f"task={getattr(model, 'task', '?')}  classes({len(labels)})={labels}")
    except Exception:
        pass

    return model


def is_stream(src):
    return str(src) == "0" or str(src).lower().endswith(VIDEO_EXT)


def build_kwargs(args):
    kw = dict(
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        verbose=False,
        save=True,                       # 保存画好框的图片/视频
        project=str(Path(args.output).parent),
        name=Path(args.output).name,
        exist_ok=True,
    )
    if args.no_txt is False:
        kw["save_txt"] = True            # 顺便让 ultralytics 写一份标准 txt
    if args.device:
        kw["device"] = args.device
    return kw


def summarize(result, args):
    """返回 (图片名, 行列表)；行列表同时打印到终端。"""
    n = 0 if result.boxes is None else len(result.boxes)
    rows = []

    name = "image"
    if getattr(result, "path", None):
        name = Path(result.path).name

    if n and result.keypoints is not None and result.keypoints.xy is not None:
        names = result.names
        xy = result.keypoints.xy.cpu().numpy()          # (n, 4, 2)
        kconf = result.keypoints.conf
        kconf = kconf.cpu().numpy() if kconf is not None else None

        for i in range(n):
            cls_id = int(result.boxes.cls[i].item())
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            cn = CLASS_CN.get(str(label), "")
            box = result.boxes.xyxy[i].cpu().numpy()
            score = float(result.boxes.conf[i].item())
            kp = xy[i]

            line = (f"{label} {score:.3f} "
                    f"{box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f}"
                    f" | kpt: " + " ".join(f"{x:.1f},{y:.1f}" for x, y in kp))
            if kconf is not None:
                line += " | kpt_conf: " + " ".join(f"{c:.2f}" for c in kconf[i])
            if args.center:
                cx, cy = kp.mean(axis=0)
                line += f" | center: {cx:.1f},{cy:.1f}"
            rows.append(line)

            info(f"  {label}{('(' + cn + ')') if cn else ''} conf={score:.3f} "
                 f"box=({box[0]:.0f},{box[1]:.0f})-({box[2]:.0f},{box[3]:.0f})"
                 + (f" center=({cx:.0f},{cy:.0f})" if args.center else ""))

    return name, n, rows


def main():
    args = parse_args()

    # 数据源默认值
    if args.source is None:
        guess = HERE / "test.jpg"
        if guess.exists():
            args.source = str(guess)
        else:
            die("没有指定数据源。请用 -s 指定，例如:\n"
                "    python infer.py -s armor.jpg\n"
                "    python infer.py -s images/\n"
                "    python infer.py -s demo.mp4\n"
                "    python infer.py -s 0          (摄像头)")
    elif str(args.source) != "0" and not Path(args.source).exists():
        die(f"数据源不存在: {args.source}")

    # 自动选设备
    if args.device is None:
        try:
            import torch
            if torch.cuda.is_available():
                args.device = "0"
                info(f"检测到 CUDA 显卡: {torch.cuda.get_device_name(0)}")
            else:
                args.device = "cpu"
                info("未检测到 CUDA，使用 CPU 推理（会慢一些）")
        except Exception:
            args.device = "cpu"

    model = load_model(args.model)

    stream = is_stream(args.source)
    info(f"数据源: {args.source}   设备: {args.device}   "
         f"conf={args.conf} iou={args.iou} imgsz={args.imgsz}")

    kwargs = build_kwargs(args)
    src = 0 if str(args.source) == "0" else str(args.source)

    total, frames = 0, 0
    save_dir = None

    try:
        results = model.predict(source=src, stream=True, **kwargs)
        for result in results:
            frames += 1
            name, n, rows = summarize(result, args)
            total += n

            if save_dir is None:
                save_dir = getattr(result, "save_dir", None)

            if n == 0:
                info(f"{name}: 没检测到装甲板")

            # 自己再写一份带关键点坐标的 txt（比 ultralytics 默认的多出 4 个角点）
            if not args.no_txt and rows:
                try:
                    if save_dir is not None:
                        txt = Path(save_dir) / (Path(name).stem + "_kpt.txt")
                        txt.parent.mkdir(parents=True, exist_ok=True)
                        txt.write_text("\n".join(rows) + "\n", encoding="utf-8")
                except Exception as e:
                    info(f"（关键点 txt 保存失败: {e}）")

            if not args.no_show and not stream:
                try:
                    result.show()
                except Exception as e:
                    info(f"（无法弹窗预览: {e}，可用 --no-show 跳过）")
                    args.no_show = True
    except KeyboardInterrupt:
        info("已中断")

    info(f"完成：{frames} 帧，共 {total} 个目标")
    if save_dir is not None:
        info(f"结果保存在: {Path(save_dir).resolve()}")
    else:
        info(f"结果保存在: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
