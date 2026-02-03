import pandas as pd

from src.preprocess import (
    filter_missing_members_for_label,
    make_hit_label,
    parse_listlike,
)


def test_filter_missing_members_for_label_removes_nulls():
    details_frame = pd.DataFrame(
        {
            "members": [100, None, 50],
            "title": ["First", "Second", "Third"],
        }
    )

    filtered_frame = filter_missing_members_for_label(details_frame)

    assert filtered_frame["members"].isna().sum() == 0
    assert len(filtered_frame) == 2


def test_make_hit_label_uses_top_fraction_threshold():
    details_frame = pd.DataFrame({"members": [10, 20, 30, 40, 50]})

    labels = make_hit_label(details_frame, topk_fraction=0.2)

    assert labels.sum() == 1
    assert labels.iloc[-1] == 1


def test_parse_listlike_splits_tokens():
    parsed_tokens = parse_listlike("Action; Adventure, Comedy")

    assert parsed_tokens == ["Action", "Adventure", "Comedy"]
