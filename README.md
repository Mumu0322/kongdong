# kongdong

YOLO + RGB-D + ChArUco + AUBO eye-in-hand hole-localization code.

Included:

- `run_yolo_eye_in_hand_optimized.py` and its backup copy;
- `aubo_workbench/` runtime modules;
- current tests, model, and calibration artifacts.

Before running on another machine, set robot connection settings instead of committing them:

```powershell
$env:AUBO_ROBOT_IP = "192.168.x.x"
$env:AUBO_ROBOT_USER = "AUBO"
$env:AUBO_ROBOT_PASSWORD = "your-password"
```

The source remains configured for the local `C:\MM` layout. Review the model, hand-eye, ChArUco report, and home-point paths before enabling robot motion.
