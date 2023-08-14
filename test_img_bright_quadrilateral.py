import pytest
import numpy as np
import img_bright_quadrilateral as ibq

@pytest.fixture
def sample_gray_image():
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)

# Test distance calculation
@pytest.mark.parametrize(
    ('p1', 'p2', 'expected_dist'),
    [
    ((0, 0), (3, 4), 5.0),
    ((1, 1), (1, 1), 0.0),
    ((2, 3), (1, 1), 2.23),
    ] 
)
def test_distance(p1: tuple, p2: tuple, expected_dist: float):
    result = ibq.distance(p1, p2)
    assert result == pytest.approx(expected_dist, abs=1e-2)


# Test 'calc_polar_angle' unit
@pytest.mark.parametrize(
    ('p1', 'p2', 'expected_angle'),
    [
    ((0, 0), (1, 1), 0.785), # positive 45 degrees
    ((0, 0), (0, 6), 1.57),  # y-axis 180 degrees
    ((2, 3), (9, 1), -0.27), # negative angle
    ((0, 0), (0, 0), 0.0),  # same points
    ] 
)
def test_calc_polar_angle(p1: tuple, p2: tuple, expected_angle: float ):
    result = ibq.calc_polar_angle(p1,p2)
    assert result == pytest.approx(expected_angle, abs=1e-2)


# Test 'order_coordinates_anticlock' unit
@pytest.mark.parametrize(
    ('points', 'expected_ordered_points'),
    [
    ([], []), # empty
    ([(3, 5)], [(3, 5)]), # single point
    ([(0, 0), (1, 1)], [(0, 0), (1, 1)]), # two points
    ([(0, 0), (2, 0), (1, 1), (0, 0)], [(0, 0), (0, 0), (2, 0), (1, 1)]), # duplicates
    ([(2, 2), (0, 2), (0, 0), (2, 0)], [(0, 0), (2, 0), (2, 2), (0, 2)]), # valid
    ]
)
def test_order_coordinates_anticlock(points:list[tuple],
                                     expected_ordered_points:list[tuple]):
    result = ibq.order_coordinates_anticlock(points)
    assert result == expected_ordered_points

# Test area calculation
@pytest.mark.parametrize(
    ['ordered_points', 'exp_area'],
    [([], None), # empty list
     ([(0, 0), (6,6)], None),  # insufficient points
     ([(0, 3), (5,0), (0, 0)], 7.5),  # three points
     ([(1,1), (1,1), (1,1), (1,1)], 0.0),  # all same points
     ([(2, 0), (2, 2), (0, 2), (0, 0)], 4.0) # valid
    ]
)
def test_calc_area(ordered_points:list, exp_area:float):
    area = ibq.calc_area(ordered_points)
    assert area == pytest.approx(exp_area, abs=1e-2)


@pytest.mark.parametrize("im_path, expected_result", [
    ("input_im.jpg", (True, True)),            # Valid jpg image
    ("lena.png", (True, True)),                 # Valid png image
    ("nonexistent_image.png", (False, False)),  # Nonexistent image
    ("abc.pdf", (False, False)),                # Invalid file format
    ("tiny.png",(False, False)),                # insufficient image size
])
def test_read_image(im_path, expected_result):
    orig_img, gray_im = ibq.read_image(im_path)

    if expected_result[0]:  # If expected to be valid
        assert isinstance(orig_img, np.ndarray) and orig_img.shape[2] == 3
        assert isinstance(gray_im, np.ndarray) and gray_im.ndim == 2
    else:  # If expected to be invalid
        assert orig_img is None
        assert gray_im is None


@pytest.mark.parametrize("patch_size, max_patch_count, expected_result", [
    (5, 4, True),  # Valid parameters
    (10, 2, True),  # Larger patch size and smaller max count
])
def test_get_allpatches_with_brightness_value(sample_gray_image: np.ndarray,
                                              patch_size: int,
                                              max_patch_count: int,
                                              expected_result: bool):
    result = ibq.get_allpatches_with_brightness_value(sample_gray_image, patch_size, max_patch_count)
    
    if expected_result:
        assert isinstance(result, list)
        for center in result:
            assert isinstance(center, list) and len(center) == 2
    else:
        assert result is None
 
        
@pytest.mark.parametrize("points, expected_result", [
    ([(0, 0), (0, 5), (5, 5), (5, 0)], True),  # Valid points
    ([(0, 0), (0, 5), (5, 5)], False),        # Insufficient points
    ([(0, 0), (0, 5), (0, 5), (5, 5)], False),  # Overlapping points
    ([(2, 2), (2, 15), (2, 30), (5, 5)], False),  # Collinear points
])
def test_draw_quadrilateral(points, expected_result):
    img = np.zeros((10, 10, 3), dtype=np.uint8)  # dummy image
    result = ibq.draw_quadrilateral(img, points)
    assert result == expected_result


if __name__ == "__main__":
    pytest.main()