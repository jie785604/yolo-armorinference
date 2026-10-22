from ultralytics import YOLO


def export_models():
    model = YOLO("weights/best.pt")

    # 导出 ONNX
    print("正在导出 ONNX 模型...")
    onnx_path = model.export(format="onnx", imgsz=640, simplify=True)
    print(f"ONNX 已保存至: {onnx_path}")

    # 导出 TorchScript
    print("正在导出 TorchScript 模型...")
    ts_path = model.export(format="torchscript", imgsz=640)
    print(f"TorchScript 已保存至: {ts_path}")


if __name__ == "__main__":
    export_models()
