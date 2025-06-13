import pytest
import pandas as pd
import numpy as np

def test_merge_preserve_keys_no_coalesce():
    df1 = pd.DataFrame({"id": [1, 2, 3], "value1": ["A", "B", "C"]})
    df2 = pd.DataFrame({"id": [2, 3, 4], "value2": ["X", "Y", "Z"]})

    res = pd.merge(
        df1, df2,
        on="id",
        how="outer",
        coalesce_keys=False,
        suffixes=("", "_right"),
    )

    expected = pd.DataFrame({
        "id": [1.0, 2.0, 3.0, np.nan], 
        "value1": ["A", "B", "C", np.nan],
        "id_right": [np.nan, 2.0, 3.0, 4.0],  
        "value2": [np.nan, "X", "Y", "Z"],
    })
    
    assert None not in res.columns
    assert res.shape == (4, 4) 

    pd.testing.assert_frame_equal(
        res.sort_index(axis=1),  
        expected.sort_index(axis=1),
        check_dtype=False,
    )
    
    
    
    
def test_merge_preserve_keys_no_coalesce_alternative():
    left = pd.DataFrame({
        "key": ["A", "B", "C", None],
        "left_val": [1, 2, 3, 4]
    })
    
    right = pd.DataFrame({
        "key": ["B", "C", "D", "E"],
        "right_val": [10, 20, 30, 40]
    })

    result = pd.merge(
        left,
        right,
        on="key",
        how="outer",
        coalesce_keys=False,
        suffixes=("_left", "_right")
    )

    # --- Verifications ---
    
    # 1. Check column names (pandas keeps original name for both key columns)
    expected_columns = ["key", "left_val", "key_right", "right_val"]
    assert list(result.columns) == expected_columns
    
    # 2. Check key columns content
    # Left key column should have original left values + NaN for right-only rows
    expected_left_key = pd.Series(["A", "B", "C", None, None], name="key")
    pd.testing.assert_series_equal(result["key"], expected_left_key)
    
    # Right key column should have original right values + NaN for left-only rows
    expected_right_key = pd.Series([None, "B", "C", "D", "E"], name="key_right")
    pd.testing.assert_series_equal(result["key_right"], expected_right_key)
    
    # 3. Check data values
    assert result["left_val"].tolist() == [1, 2, 3, 4, None]
    assert result["right_val"].tolist() == [None, 10, 20, 30, 40]
    
    # 4. Check shape (5 rows × 4 columns)
    assert result.shape == (5, 4)
