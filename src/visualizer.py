import cv2


def draw_tracks(frame, tracks, speeds, calibrated=False):
    h_img, w_img = frame.shape[:2]
    scale = max(0.55, min(1.6, w_img / 1920.0))
    box_thick = max(2, int(2 * scale))
    font_scale = 0.55 * scale
    font_thick = max(1, int(1.8 * scale))

    for t in tracks:
        if t.missed != 0:
            continue
        x, y, w, h = t.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), box_thick)
        label = f"ID {t.track_id} | {t.class_name} | {t.confidence:.2f}"
        cv2.putText(
            frame, label, (x, max(int(24 * scale), y - int(8 * scale))),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 200, 0), font_thick, cv2.LINE_AA
        )
        for a, b in zip(t.history[:-1], t.history[1:]):
            cv2.line(frame, a, b, (255, 0, 0), box_thick)

        speed = speeds.get(t.track_id, 0.0)
        if calibrated:
            speed_text = f"{speed * 3.6:.1f} km/h"
        else:
            speed_text = f"{speed:.1f} px/s"

        cv2.putText(
            frame, speed_text, (x, y + h + int(20 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (0, 230, 50), font_thick, cv2.LINE_AA
        )
    return frame


def draw_hud(frame, frame_no, active, counts, calibrated):
    h_img, w_img = frame.shape[:2]
    scale = max(0.55, min(1.6, w_img / 1920.0))
    font_scale = 0.65 * scale
    font_thick = max(1, int(2 * scale))
    line_spacing = int(32 * scale)

    calib_str = "Speed: Calibrated (Road Homography -> km/h)" if calibrated else "Speed: Pixel displacement (px/s)"
    lines = [
        f"Frame: {frame_no}",
        f"Active Vehicles: {active}",
        "Class Breakdown: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) if counts else "None"),
        calib_str,
    ]

    # Draw semi-transparent background plate for readability
    plate_w = int(680 * scale)
    plate_h = int((len(lines) * line_spacing) + 20 * scale)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + plate_w, 10 + plate_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = int(38 * scale)
    for line in lines:
        cv2.putText(frame, line, (int(20 * scale), y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
        y += line_spacing
    return frame
