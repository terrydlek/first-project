import pandas as pd
import glob

remnant_file = glob.glob(r"C:\Users\USER\PycharmProjects\pythonProject3\전산 프로젝트\엑셀에서 읽어 재고 파악하기\\재고_*.xlsx")

print(remnant_file)

merge_df = pd.DataFrame()

for file in remnant_file:
    df_from_excel = pd.read_excel(file)
    df_from_excel['재고위치'] = file.split(".")[0]
    merge_df = pd.concat([merge_df, df_from_excel])

print(merge_df)

filter_df = merge_df[merge_df['날짜'] < '2015-01-01']

print(filter_df)

filter_df = merge_df[merge_df["날짜"].between('2012-1-1', '2015-12-13')]
print(filter_df)

filter_df = filter_df[filter_df['수량'] < 15]
filter_df.to_excel('날짜_수량.xlsx')
print(filter_df)