import pytest
import lexi_quadrilateral as lq


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
    result = lq.distance(p1, p2)
    assert result == pytest.approx(expected_dist, abs=1e-2)


# Test 'calc_polar_angle' unit
@pytest.mark.parametrize(
    ('p1', 'p2', 'expected_angle'),
    [
    ((0, 0), (1, 1), 0.785),
    ((0, 0), (0, 6), 1.57),
    ((2, 3), (9, 1), -0.27),
    ((0, 0), (0, 0), 0.0),
    ] 
)
def test_calc_polar_angle(p1: tuple, p2: tuple, expected_angle: float ):
    result = lq.calc_polar_angle(p1,p2)
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
    result = lq.order_coordinates_anticlock(points)
    assert result == expected_ordered_points


# Test fetch_highlexi_str_with_loc functionality
@pytest.mark.parametrize(
    ('txt_file_path', 'expected_result'),
    ( 
     ["empty.txt", []], # empty-file
     ["demo1.txt",[('omput', 0, 29), ('ch110', 0, 19)]], # not enough substrings/points
     ["demo2.txt",[('zzzzz', 3, 26), ('zzzzz', 11, 8),('zzzzz', 13, 64), ('zzzzz', 17, 19)]], # all-substrings equal values
     ["demo3.txt",[('zzzzz', 4, 0), ('yy6dv', 4, 6), ('xgp0z', 4, 12), ('wvTQ4', 4, 18)]], # co-linear substring points
     ["demo4.txt", [('zzzzz', 17, 19), ('ython', 1, 9), ('xwvu8', 20, 0), ('xicog', 2, 2)]], # unicodes in file
     ["demo5.txt", [('zzzzz', 13, 47), ('zzzzz', 13, 52), ('ython', 1, 9), ('xwvu8', 19, 0)]], # adjacent substrings points
    )
)
def test_fetch_highlexi_str_with_loc(txt_file_path: str, expected_result:list):
    WORD_LENGTH = 5
    WORD_COUNT = 4
    file_content = lq.read_textfile(txt_file_path)
    result = lq.fetch_highlexi_str_with_loc(file_txt=file_content,
                                            word_length=WORD_LENGTH,
                                            word_count=WORD_COUNT)
    assert result == expected_result


# Test perimeter calculation
@pytest.mark.parametrize(
    ['ordered_points', 'exp_perimeter'],
    [([], None), # empty list
     ([(0, 0), (6,6)], None),  # insufficient points
     ([(0, 3), (4, 3), (1,1)], 9.84),  # three points
     ([(1,1), (1,1), (1,1), (1,1)], 0.0),  # all same points
     ([(0, 0), (0, 3), (4, 3), (4, 0)], 14.0) # valid
    ]
)
def test_calc_perimeter(ordered_points:list, exp_perimeter:float):
    perimeter = lq.calc_perimeter(ordered_points)
    assert perimeter == pytest.approx(exp_perimeter, abs=1e-2)


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
    area = lq.calc_area(ordered_points)
    assert area == pytest.approx(exp_area, abs=1e-2)


# smoke test
@pytest.mark.parametrize(
    ['txt_file_path', 'exp_perimeter', 'exp_area'],
    [('empty.txt', None, None), # empty file
     ('demo.txt', 63.28, 229.0), # valid file
    ]
)
def test_lexi_quadrilateral(txt_file_path:str, exp_perimeter:float, exp_area:float ):
    WORD_LENGTH = 5
    WORD_COUNT = 4
    file_content = lq.read_textfile(txt_file_path)
    topk_substrings_with_loc = lq.fetch_highlexi_str_with_loc(file_txt=file_content,
                                            word_length=WORD_LENGTH,
                                            word_count=WORD_COUNT)
    coordinates = [(i,j) for _, i, j in topk_substrings_with_loc]
    ordered_points = lq.order_coordinates_anticlock(coordinates)
    peri = lq.calc_perimeter(ordered_points)
    area = lq.calc_area(ordered_points)

    assert peri == pytest.approx(exp_perimeter, abs=1e-2)
    assert area == pytest.approx(exp_area, abs=1e-2)   


if __name__ == "__main__":
    pytest.main()
