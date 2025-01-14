from datamop.column_encoder import column_encoder 
import pandas as pd

def test_column_encoder():
    df = pd.DataFrame({
        'Sport': ['Tennis', 'Basketball', 'Football', 'Badminton'],
        'Level': ['A', 'B', 'C', 'D']
    })
    result_df_one = column_encoder(df, columns=['Sport'], method='one-hot')
    expected_columns_one = ['Level', 'Sport_Badminton', 'Sport_Basketball', 'Sport_Football', 'Sport_Tennis']
    
    assert list(result_df_one.columns) == expected_columns_one
    assert result_df_one.shape == (4, 5)  # 4 rows, 5 columns
    assert result_df_one['Sport_Tennis'].tolist() == [1, 0, 0, 0]
    assert result_df_one['Sport_Basketball'].tolist() == [0, 1, 0, 0]
    
    custom_order = {'Level': ['A', 'B', 'C', 'D']}
    result_df_ord = column_encoder(df, columns=['Level'], method='ordinal', order=custom_order)
    
    expected_columns_ord = ['Sport', 'Level']
    assert list(result_df_ord.columns) == expected_columns_ord
    assert result_df_ord['Level'].tolist() == [0, 1, 2, 3]