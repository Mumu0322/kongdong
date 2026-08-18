from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    model_path = Path(__file__).with_name("best.pt")
    output_path = YOLO(model_path).export(
        format="engine",
        half=True,
        device=0,
        imgsz=[1280, 1088],
        project=str(model_path.parent),
        name="yolo26-seg-v1_new",
        verbose=True,
    )
    print(f"导出完成，文件位于: {output_path}")


if __name__ == "__main__":
    main()
