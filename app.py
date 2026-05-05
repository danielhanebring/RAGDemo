import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Dummy data 
texts = [
    "Semester ansöks via HR-systemet.",
    "Arbetstid är 40 timmar per vecka.",
    "Lön betalas ut i efterskott.",
    "Sjukfrånvaro rapporteras i HR-systemet."
]

# Skapar en text-splitter
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
#Omvandlar text till dokument object (Varje text blir ett dokument)
docs = splitter.create_documents(texts)

# Skapar en embedding-modell gör om mina dokument till vektorer
embeddings = HuggingFaceEmbeddings()

# Alla "dokument" blir till embeddings och sparas
db = Chroma.from_documents(docs, embeddings)

# UI
st.title("Retrieval Demo")

# UI
query = st.text_input("Ställ en fråga")

#När användaren skriver något körs retrieval
if query:
    #Frågan blir embedding och jämförs med alla "dokument", räknar likheter och returnerar den bästa matchen
    results = db.similarity_search(query, k=1)

    st.write("Mest relevanta dokumentet:")

#Visar innehållet i det mest relevanta "dokumentet"
    for doc in results:
        st.write("- " + doc.page_content)