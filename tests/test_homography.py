from src.homography import HomographyTransformer


def test_identity_like_mapping():
    t = HomographyTransformer(
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        [[0, 0], [10, 0], [10, 10], [0, 10]],
    )
    x, y = t.transform_point((5, 5))
    assert abs(x - 5) < 1e-4 and abs(y - 5) < 1e-4
