import cv2
import os
from ultralytics import YOLO

def main():
    # 加载模型
    model = YOLO('weights/best.pt') 

    # 视频路径
    video_path = 'data/test_video.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频，请检查路径: {video_path}")
        return

    os.makedirs('outputs', exist_ok=True)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('outputs/result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        annotated_frame = results[0].plot() # 自动绘制检测框和关键点
        
        out.write(annotated_frame)
        cv2.imshow('YOLO Pose Inference', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("推理完成！结果已保存至 outputs/result.mp4")

if __name__ == '__main__':
    main()