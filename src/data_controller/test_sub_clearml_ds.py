import random
from collections import defaultdict

import pytest
from src.data_controller.sub_clearml_ds import (__query_dataset_id,
                                                __regex_input)


def test_regex_input(dataset_input, expected_dataset_id, expected_limit_raw):
    dataset_id, limit_raw = __regex_input(dataset_input)
    assert dataset_id == expected_dataset_id
    assert limit_raw == expected_limit_raw


@pytest.mark.parametrize(
    "dataset_input, expected_dataset_id, expected_limit_raw",
    [
        (
            "f580a16f0d8e466d9ec79d12d15e071d|[-*:all, overripe:all]",
            "f580a16f0d8e466d9ec79d12d15e071d",
            ["-*:all", "overripe:all"],
        ),
        ("abcd1234|[ripe:5]", "abcd1234", ["ripe:5"]),
        ("xyz9876|[*]", "xyz9876", ["*"]),
        ("xyz9876  | [*]", "xyz9876", ["*"]),
    ],
)
def test_regex_input(dataset_input, expected_dataset_id, expected_limit_raw):
    dataset_id, limit_raw = __regex_input(dataset_input)
    assert dataset_id == expected_dataset_id
    assert limit_raw == expected_limit_raw


d_urls_by_cat_test = {
    "class1": ["url1"] * 100,
    "class2": ["url1"] * 45,
    "class3": ["url1"] * 65,
}


@pytest.mark.parametrize(
    "limit_raw, d_urls_by_cat, expected_output",
    [
        (
            ["-*:all", "class2:all"],
            d_urls_by_cat_test,
            {
                # 'class1': len(d_urls_by_cat_test['class1']),
                "class2": len(d_urls_by_cat_test["class2"])
                # 'class3': len(d_urls_by_cat_test['class3'])
            },
        ),
        (
            ["*:all", "class2:10", "-class1:6"],
            d_urls_by_cat_test,
            {
                "class1": len(d_urls_by_cat_test["class1"]) - 6,
                "class2": 10,
                "class3": len(d_urls_by_cat_test["class3"]),
            },
        ),
        # (
        #     ['ripe:2'], d_urls_by_cat_test,
        #     {
        #         'class1': len(d_urls_by_cat_test['class1']),
        #         'class2': len(d_urls_by_cat_test['class2']),
        #         'class3': len(d_urls_by_cat_test['class3'])
        #     }
        # ),
        # (
        #     ['*'], d_urls_by_cat_test,
        #     {
        #         'class1': len(d_urls_by_cat_test['class1']),
        #         'class2': len(d_urls_by_cat_test['class2']),
        #         'class3': len(d_urls_by_cat_test['class3'])
        #     }
        # )
    ],
)
def test_query_dataset_id(limit_raw, d_urls_by_cat, expected_output):
    d_new_urls_by_cat = __query_dataset_id(limit_raw, d_urls_by_cat)
    d_count = {k: len(v) for k, v in d_new_urls_by_cat.items()}
    assert d_count == expected_output


# # Run the tests
# pytest.main()
