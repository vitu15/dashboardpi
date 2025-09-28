import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# ---------------------------
# Dados simulados
# ---------------------------
dados = pd.DataFrame({
    "Oficina": ["Auto Center Goiás", "Mecânica Rápida", "Oficina 10", "Top Motors", "Mecânica do João"],
    "Avaliação": [4.5, 3.8, 4.2, 4.7, 3.9],
    "Serviços Realizados": [120, 90, 150, 200, 80]
})

chamados = pd.DataFrame({
    "Sintoma": [
        "Carro não liga", "Freio fazendo barulho", "Motor superaquecendo",
        "Pneu furado", "Luz do óleo acesa", "Bateria descarregada", 
        "Carro não dá partida", "Barulho estranho no freio", "Radiador vazando"
    ],
    "Categoria": [
        "Elétrico", "Freios", "Arrefecimento",
        "Pneus", "Motor", "Elétrico",
        "Elétrico", "Freios", "Arrefecimento"
    ]
})

# ---------------------------
# Treinando IA (NLP + Naive Bayes)
# ---------------------------
modelo = make_pipeline(TfidfVectorizer(), MultinomialNB())
modelo.fit(chamados["Sintoma"], chamados["Categoria"])

# ---------------------------
# Dashboard
# ---------------------------
st.set_page_config(page_title="Mechaniker Dashboard", layout="wide")
st.title("Mechaniker - Dashboard com IA Aprimorada")

menu = st.sidebar.radio("Navegação", ["Visão Geral", "Oficinas", "Chamados", "IA de Predição"])

# ---------------------------
# Aba 1 - Visão Geral
# ---------------------------
if menu == "Visão Geral":
    col1, col2, col3 = st.columns(3)
    col1.metric("Oficinas Cadastradas", len(dados))
    col2.metric("Chamados Registrados", len(chamados))
    col3.metric("Média Avaliações", round(dados["Avaliação"].mean(), 2))

    fig = px.bar(dados, x="Oficina", y="Serviços Realizados", color="Serviços Realizados",
                 title="Serviços Realizados por Oficina")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Aba 2 - Oficinas
# ---------------------------
elif menu == "Oficinas":
    st.subheader("Avaliações das Oficinas")
    filtro = st.slider("Filtrar por avaliação mínima", 0.0, 5.0, 3.5)
    st.write(dados[dados["Avaliação"] >= filtro])

    fig = px.bar(dados[dados["Avaliação"] >= filtro], 
                 x="Avaliação", y="Oficina", orientation="h", color="Avaliação",
                 title="Avaliações por Oficina")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Aba 3 - Chamados
# ---------------------------
elif menu == "Chamados":
    st.subheader("Chamados Registrados")
    st.write(chamados)

    fig = px.histogram(chamados, x="Categoria", title="Distribuição de Chamados por Categoria")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Aba 4 - IA de Predição
# ---------------------------
elif menu == "IA de Predição":
    st.subheader("Predição de Categoria de Problema")
    sintoma = st.text_input("Digite o sintoma do veículo:", "")
    
    if sintoma:
        pred = modelo.predict([sintoma])[0]
        st.success(f"Categoria prevista: **{pred}**")
