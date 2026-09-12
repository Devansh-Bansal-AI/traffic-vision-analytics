from src.speed_estimator import SpeedEstimator


def test_pixel_speed_positive():
    e = SpeedEstimator(fps=10)
    assert e.update(1, (0, 0)) == 0
    assert e.update(1, (10, 0)) > 0
