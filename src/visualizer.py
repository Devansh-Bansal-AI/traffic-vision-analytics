import cv2


def draw_tracks(frame, tracks, speeds, calibrated=False):
    for t in tracks:
        if t.missed != 0:
            continue
        x, y, w, h = t.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
        label = f"ID {t.track_id} | {t.class_name} | {t.confidence:.2f}"
        cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 200, 0), 2, cv2.LINE_AA)
        for a, b in zip(t.history[:-1], t.history[1:]):
            cv2.line(frame, a, b, (255, 0, 0), 2)
        speed = speeds.get(t.track_id, 0.0)
        unit = "m/s" if calibrated else "px/s"
        cv2.putText(frame, f"{speed:.1f} {unit}", (x, y + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2, cv2.LINE_AA)
    return frame


def draw_hud(frame, frame_no, active, counts, calibrated):
    lines = [
        f"Frame: {frame_no}",
        f"Active vehicles: {active}",
        "Classes: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "Speed: calibrated" if calibrated else "Speed: pixel displacement (calibration required for m/s)",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += 25
    return frame
