# Video Data Directory

Place your traffic video files in this directory (`data/`).

### Example
```text
data/18437773-uhd_3840_2160_50fps.mp4
```

### Note on Large Files
Raw 4K / UHD video recordings are large (>100MB) and are excluded from Git version control via `.gitignore`.
For running the analysis, ensure your input video path is passed via the `--input` flag:

```bash
python main.py --input data/18437773-uhd_3840_2160_50fps.mp4 --max-frames 300
```
