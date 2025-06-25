import streamlit as st
import requests

import pandas as pd
import matplotlib.pyplot as plt

def request_predict(text: str):
    return requests.post("http://127.0.0.1:8002/predict", params={"text": text})

def main():
    page = st.sidebar.radio("Меню", ["Предсказание", "Статистика"])

    if page == "Предсказание":
        st.title("Предсказание тематики статей")
        input_text = st.text_area("Введите содержимое", height=150, placeholder="Содержимое статьи")

        if st.button("Определить"):
            if input_text.strip():
                try:
                    response = request_predict(input_text)
                    response.raise_for_status()
                    result = response.json()

                    probabilities = result["probabilities"]
                    sorted_probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

                    try:
                        for comment_nature, probability in sorted_probabilities:
                            st.write(f"{comment_nature} - {probability:.3f} %")

                    except Exception as e:
                        st.error(f"Ошибка обработки вероятностей: {e}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Ошибка соединения с API: {e}")
            else:
                st.warning("Пожалуйста, введите текст")

    else:
        df = pd.read_csv("Habr (Cluster).csv", keep_default_na=False)
        total_len = len(df)

        st.markdown(f"Общая длина набора данных: {total_len}")
        st.markdown(f"")

        st.markdown(f"Распределение статей:")
        counts = df["cluster name"].value_counts()

        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        ax.set_xticklabels(counts.index, rotation=-90)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
