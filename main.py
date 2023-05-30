import os
import io
import streamlit as st
from streamlit_chat import message
import pandas as pd

from langchain import OpenAI
# from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.schema import HumanMessage

### 環境変数設定
## ローカル実行の場合
# from dotenv import load_dotenv
# load_dotenv()
## Sreamlit Cloudにデプロイする場合
os.environ['OPENAI_API_KEY'] = st.secrets.OpenAIAPI.openai_api_key

### モデル設定
## ローカル実行の場合
# model_name="gpt-3.5-turbo"
# model_name="text-davinci-003"
## Sreamlit Cloudにデプロイする場合
model_name=st.secrets.OpenAIModel.model_name

df = pd.DataFrame()
# 画面UI定義-------------------------------------------------------------------------------------
# Streamlitによって、タイトル部分のUIをの作成
st.title("CSV データ検索")

# ファイルアップローダーの作成
uploaded_file = st.file_uploader("Choose a file", type='csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

# 入力フォームと送信ボタンのUIの作成
text_input = st.text_input("検索したい情報を入力してください。")
# チェックボックスの作成
col1, col2 = st.columns([1, 7]) 
with col1:
    search_button = st.button("検索")
with col2:
    is_table = st.checkbox("テーブル形式で出力")
# -------------------------------------------------------------------------------------------

# ChatGPT-3.5のモデルのインスタンスの作成
llm = OpenAI(model_name=model_name, temperature=0.2)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
agent = create_pandas_dataframe_agent(llm, df, verbose=True)

# 実行ボタンが押された場合
if search_button:
    search_button = False

    if not uploaded_file:
        # ファイルがアップロードされていない場合、エラーメッセージを出力
        message('CSVファイルをアップロードしてください。', is_user=False, key=9999, avatar_style="bottts-neutral", seed='Felix')
    elif not text_input:
        # テキストボックスの入力がない場合、エラーメッセージを出力
        message('検索したい情報を正しく入力してください。', is_user=False, key=9999, avatar_style="bottts-neutral", seed='Felix')
    elif is_table:
        # 「テーブル形式で出力」チェックボックスがチェックされている場合の検索処理
        final_input = text_input + ' Output it to a CSV file with headers named output.csv.'
        agent.run(final_input)
        df_data = pd.read_csv('output.csv')
        print("df_data:",df_data)
        print("df_data type", type(df_data))
        # テーブル形式で取得結果を表示
        st.table(df_data)
    else:
        #「テーブル形式で出力」チェックボックスがチェックされていない場合の検索処理
        final_input = text_input + ' Give it back in japanese.'
        # final_input = text_input
        output = agent.run(final_input)
        # メッセージ形式で取得結果を表示
        message(output, is_user=False, key=1, avatar_style="bottts-neutral", seed='Aneka')
