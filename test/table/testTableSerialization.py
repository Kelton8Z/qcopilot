import pandas as pd
def load_data(filename: str, extra_info: dict = None):
    df = pd.read_excel(filename, sheet_name=None)
    print(df)
    sentences = []
    # if {'Model', 'Input Length', 'Output Length', 'Batch Size', 'Latency'}.issubset(df.columns):
    for sheetname, sheet in df.items():
        cols = sheet.columns
        for row in sheet.itertuples(index=False, name=None):
            sentence = f"For {sheetname}, "
            for cell, col in zip(row, cols):
                if pd.notna(cell):
                    sentence += f'{col} is {cell}, '
                    
            sentences.append(sentence)
            
    print(sentences)
            
excel_file_path = 'sheets/模型性能对比.xlsx'
load_data(excel_file_path)