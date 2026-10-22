import cv2
import json
import os
import time

import torch
from ultralytics import YOLO


def resolve_device():
    """有 CUDA 就用 GPU，否则回退到 CPU。"""
    return 0 if torch.cuda.is_available() else "cpu"


def warn_onnx_provider(device):
    """ONNX 若没有 CUDA provider 会退回 CPU，和 GPU 上的 .pt 对比就不公平了。"""
    if device == "cpu":
        return
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception as exc:  # pragma: no cover
        print(f"警告：无法检查 onnxruntime provider（{exc}）")
        return
    if "CUDAExecutionProvider" not in providers:
        print(
            "警告：onnxruntime 未检测到 CUDAExecutionProvider，"
            "ONNX 将使用 CPU 推理，与 GPU 上的 .pt 对比不公平。"
        )


def extract_result(result):
    """把 Ultralytics 的 Results 转成可序列化的字典。"""
    boxes = result.boxes
    return {
        "boxes": boxes.xyxy.cpu().numpy().tolist(),
        "conf": boxes.conf.cpu().numpy().tolist(),
        "cls": boxes.cls.cpu().numpy().astype(int).tolist(),
        "kpts": (
            result.keypoints.xy.cpu().numpy().tolist()
            if result.keypoints is not None
            else []
        ),
    }


def compare_against(base, other):
    """以 base 为基准，按检测输出顺序统计框/关键点的一致性。"""
    if not base or not other:
        return None

    stats = {
        "frames": 0,
        "count_mismatch_frames": 0,
        "box_max_diff_px": 0.0,
        "kpt_max_diff_px": 0.0,
    }

    for a, b in zip(base, other):
        stats["frames"] += 1
        if len(a["boxes"]) != len(b["boxes"]):
            stats["count_mismatch_frames"] += 1
            continue
        if not a["boxes"]:
            continue

        box_a = torch.tensor(a["boxes"], dtype=torch.float32)
        box_b = torch.tensor(b["boxes"], dtype=torch.float32)
        stats["box_max_diff_px"] = max(
            stats["box_max_diff_px"], float((box_a - box_b).abs().max())
        )

        kpt_a = torch.tensor(a["kpts"], dtype=torch.float32)
        kpt_b = torch.tensor(b["kpts"], dtype=torch.float32)
        if kpt_a.numel() and kpt_a.shape == kpt_b.shape:
            stats["kpt_max_diff_px"] = max(
                stats["kpt_max_diff_px"], float((kpt_a - kpt_b).abs().max())
            )

    return stats


def main():
    # 1. 配置模型字典
    model_paths = {
        "pt": "weights/best.pt",
        "onnx": "weights/best.onnx",
        "torchscript": "weights/best.torchscript",
    }

    # 2. 配置输入和输出路径
    video_path = "data/test_video.mp4"
    output_dir = "results"  # 统一保存对比结果
    os.makedirs(output_dir, exist_ok=True)

    device = resolve_device()
    print(f"推理设备: {'GPU (cuda:0)' if device == 0 else 'CPU'}")

    # 提前检查视频是否能打开
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频，请检查路径: {video_path}")
        print("提示：data/test_video.mp4 因体积原因未提交到仓库，请自行准备测试视频。")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"视频信息: {width}x{height}, {fps} FPS, 共 {total_frames} 帧\n")

    collected = {}
    timing = {}

    # 3. 遍历三个模型进行对比测试
    for fmt, model_path in model_paths.items():
        print(f"========== 正在测试格式: {fmt.upper()} ==========")

        if not os.path.exists(model_path):
            print(f"找不到模型文件: {model_path}，跳过...")
            continue

        if fmt == "onnx":
            warn_onnx_provider(device)

        # 加载模型
        model = YOLO(model_path)

        # 打开视频和创建视频写入器
        cap = cv2.VideoCapture(video_path)
        out_video_path = os.path.join(output_dir, f"result_{fmt}.mp4")
        out = cv2.VideoWriter(
            out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not out.isOpened():
            print(f"无法创建输出视频: {out_video_path}（请检查 OpenCV 的 mp4 编码器）")
            cap.release()
            continue

        frame_count = 0
        total_inference_time = 0.0
        records = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 预热：跳过前 10 帧不记录时间（让模型和硬件准备好）
            if frame_count < 10:
                model(frame, verbose=False, device=device)
                frame_count += 1
                continue

            # 开始计时（只计算模型推理的时间，不包含视频读取和保存的时间）
            start_t = time.perf_counter()
            results = model(frame, verbose=False, device=device)
            end_t = time.perf_counter()

            total_inference_time += end_t - start_t

            result = results[0]
            records.append({"frame": frame_count, **extract_result(result)})

            annotated_frame = result.plot()
            out.write(annotated_frame)

            # 如果你想实时看画面，取消下面两行的注释（按 q 退出）
            # cv2.imshow(f'YOLO {fmt}', annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break

            frame_count += 1
            if frame_count % 50 == 0:
                print(f"已处理 {frame_count}/{total_frames} 帧...")

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        collected[fmt] = records

        # 计算平均耗时
        valid_frames = frame_count - 10
        if valid_frames > 0:
            avg_time_ms = (total_inference_time / valid_frames) * 1000
            timing[fmt] = avg_time_ms
            print(f"【{fmt.upper()}】 测试完毕！")
            print(f"平均每帧推理耗时: {avg_time_ms:.2f} ms")
            print(f"视频已保存至: {out_video_path}\n")
        else:
            print("视频太短，未能有效测速。\n")

    # 4. 以 .pt 为基准，比较导出模型的数值差异
    comparison = {}
    base = collected.get("pt")
    if base:
        print("========== 数值一致性对比（基准: .pt） ==========")
        for fmt in ("onnx", "torchscript"):
            stats = compare_against(base, collected.get(fmt))
            if stats is None:
                continue
            comparison[fmt] = stats
            print(
                f"[{fmt}] 对比帧数 {stats['frames']}，"
                f"检测数不一致 {stats['count_mismatch_frames']} 帧，"
                f"框最大偏差 {stats['box_max_diff_px']:.3f} px，"
                f"关键点最大偏差 {stats['kpt_max_diff_px']:.3f} px"
            )
    else:
        print("未找到 .pt 模型，跳过数值一致性对比。")

    # 5. 保存统计结果，便于写实验报告和复现
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video": video_path,
                "device": "cuda:0" if device == 0 else "cpu",
                "avg_inference_ms": timing,
                "comparison_vs_pt": comparison,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n统计结果已保存至: {metrics_path}")


if __name__ == "__main__":
    main()
