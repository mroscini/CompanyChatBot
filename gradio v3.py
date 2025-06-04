import json
import numpy as np
import faiss
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import psutil
import os
import psutil
import threading
import time
import GPUtil  # 
import torch



process = psutil.Process(os.getpid())

 # --- CONFIG ---
JSON_PATH = ["manuale rag.json","faq_annidate_rag.json"]     # Il tuo file JSON
EMBEDDINGS_SAVE_PATH = "embeddings.index"
METADATA_SAVE_PATH = "metadata.json"
TOP_K = 4

def distanza_cosine(a, b):
    return cosine(a, b)

def aggrega_sezioni_cosine(test_list, embedder, soglia_aggregazione=0.4):
    embeddings = embedder.embed_documents(test_list)
    
    aggregato = [test_list[0]]
    ultimo_embedding = embeddings[0]
    
    for i in range(1, len(test_list)):
        dist = distanza_cosine(embeddings[i], ultimo_embedding)
        if dist < soglia_aggregazione:
            aggregato[-1] += "\n\n" + test_list[i]
            # Aggiorna embedding medio (facoltativo)
            ultimo_embedding = (np.array(ultimo_embedding) + np.array(embeddings[i])) / 2
        else:
            aggregato.append(test_list[i])
            ultimo_embedding = embeddings[i]
    return aggregato



# --- 1. Carica sezioni dal JSON ---
def carica_sezioni_da_json(json_path):
    def estrai_sezioni_da_entry(entry, prefix=""):
        sezioni = []
        titoli = []

        titolo_corrente = entry.get("title", "")
        titolo_completo = f"{prefix} > {titolo_corrente}" if prefix else titolo_corrente

        testo = entry.get("text_content")
        if testo and testo.strip():
            titoli.append(titolo_completo)
            sezioni.append(f"{titolo_completo}\n{testo.strip()}")

        for sub in entry.get("subsections", []):
            sub_sezioni, sub_titoli = estrai_sezioni_da_entry(sub, titolo_completo)
            sezioni.extend(sub_sezioni)
            titoli.extend(sub_titoli)

        return sezioni, titoli

    tutte_sezioni = []
    tutti_titoli = []

    for path in json_path:
        print(f"Carico: {path}")
        with open(path, "r", encoding="utf-8") as f:
            sezioni_raw = json.load(f)

        for entry in sezioni_raw:
            sezioni, titoli = estrai_sezioni_da_entry(entry)
            tutte_sezioni.extend(sezioni)
            tutti_titoli.extend(titoli)

    return tutte_sezioni, tutti_titoli

# --- 2. Costruzione indice ---
def crea_e_salva_indice(testi, embedder, embeddings_path, metadata_path):
    embeddings = embedder.embed_documents(testi)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, embeddings_path)
    metadata = [{"testo": t} for t in testi]
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return index

# --- 3. Similarity search ---
def similarity_search(index, metadata_path, query_embedding, k=10):
    D, I = index.search(np.array([query_embedding]).astype("float32"), k)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    risultati = [meta[i]["testo"] for i in I[0] if i < len(meta)]
    return risultati

def conversational_llm(question,llm):
    prompt = f"""
    Sei un assistente intelligente Devi rispondere con "SI" se la seguente frase è un saluto, una forma di cortesia o una frase conversazionale (es. ciao, grazie, arrivederci), altrimenti rispondi con "NO".

    Frase:
    {question}
    """
    response = llm.invoke(prompt)
    risposta = response.content.strip().upper()
    return risposta


# --- 4. LLM wrapper ---
class OllamaLLM:
    def __init__(self, model_name="mistral-nemo", temperature=0.1):
        self.llm = ChatOllama(model=model_name, temperature=temperature)
    
    def invoke(self, prompt):
        return self.llm.invoke(prompt)
    

class ResourceMonitor:
    def __init__(self):
        self.max_ram = 0
        self.max_cpu = 0
        self.max_gpu = 0
        self._stop = False
        self.thread = threading.Thread(target=self._monitor)
    
    def _monitor(self):
        process = psutil.Process()
        while not self._stop:
            try:
                ram = process.memory_info().rss / (1024 ** 2)
                cpu = psutil.cpu_percent(interval=0.1)
                gpu = GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0

                self.max_ram = max(self.max_ram, ram)
                self.max_cpu = max(self.max_cpu, cpu)
                self.max_gpu = max(self.max_gpu, gpu)
            except Exception:
                pass
            time.sleep(0.05)

    def start(self):
        self._stop = False
        self.thread.start()

    def stop(self):
        self._stop = True
        self.thread.join()
        return {
            "RAM_MB": self.max_ram,
            "CPU_%": self.max_cpu,
            "GPU_%": self.max_gpu,
        }

llm = OllamaLLM()

# --- 5. Reranking ---
def rerank_documents(question, candidate_texts):
    prompt = f"""
Sei un esperto assistente. Ordina i seguenti documenti dal più rilevante al meno rilevante per rispondere alla domanda.

Domanda:
{question}

Documenti:
"""
    prompt += "\n".join([f"{i+1}. {d[:200].replace('\n',' ')}..." for i, d in enumerate(candidate_texts)])
    prompt += "\nRispondi con i numeri dei documenti ordinati separati da virgola, es: 3,1,2"

    response = llm.invoke(prompt)
    ordine = response.content.strip()

    try:
        indici = [int(x.strip())-1 for x in ordine.split(",")]
    except:
        indici = list(range(len(candidate_texts)))

    return [candidate_texts[i] for i in indici if i < len(candidate_texts)]

# --- 6. Prompt finale per risposta ---
def genera_risposta(context, question):
    prompt = f"""
    Sei un assistente esperto. Usa il contesto per rispondere chiaramente e brevemente alla domanda.
    Contesto:
    {context}

    Domanda:
    {question} in agricolus 
    """
    response = llm.invoke(prompt)
    return response.content

# --- 7. Funzione principale ---
def answer_question_with_fallback_llm(question, history):
    monitor = ResourceMonitor()
    monitor.start() # MB prima
    a=conversational_llm(question,llm)
    torch.cuda.empty_cache() 
    metrics_after_conversational = monitor.stop()
    print(f"[USO DOPO conversational_llm] RAM: {metrics_after_conversational['RAM_MB']:.2f} MB | "
          f"CPU: {metrics_after_conversational['CPU_%']:.1f}% | GPU: {metrics_after_conversational['GPU_%']:.1f}%")
    if a == "SI":
        conv=llm.invoke(question).content
        return conv
    else:
        # altrimenti esegui la ricerca come prima
        query_embedding = embedder.embed_query(question)
        D, I = index.search(np.array([query_embedding]).astype("float32"), k=1)
        distanza_minima = D[0][0]

        SOGLIA_DISTANZA = 1.1 # da tarare

        if distanza_minima > SOGLIA_DISTANZA:
            return "Mi dispiace, non ho informazioni su questo argomento. Prova a chiedere qualcosa riguardo il manuale."

        candidates = similarity_search(index, METADATA_SAVE_PATH, query_embedding, k=10)
        monitor = ResourceMonitor()
        monitor.start() 
        reranked = rerank_documents(question, candidates)
        torch.cuda.empty_cache()
        # Aggregazione semantica per unire sezioni simili
        metrics_after_rank = monitor.stop()
        print(f"[USO DOPO rank] RAM: {metrics_after_rank['RAM_MB']:.2f} MB | "
          f"CPU: {metrics_after_rank['CPU_%']:.1f}% | GPU: {metrics_after_rank['GPU_%']:.1f}%")
        aggregati = aggrega_sezioni_cosine(reranked, embedder, soglia_aggregazione=0.4)

        # Prendo i primi TOP_K aggregati
        top_contexts = aggregati[:TOP_K]

        context = "\n\n".join(top_contexts)
        monitor = ResourceMonitor()
        monitor.start() 
        # Genero la risposta finale con il contesto aggregato
        answer = genera_risposta(context, question)
        metrics_risposta = monitor.stop() 
        torch.cuda.empty_cache() # Stop e ottieni picchi
        print(f"[USO PICCO] RAM: {metrics_risposta['RAM_MB']:.2f} MB | CPU: {metrics_risposta['CPU_%']:.1f}% | GPU: {metrics_risposta['GPU_%']:.1f}%")

        
        return answer

# --- 8. MAIN EXECUTION ---
print("Caricamento sezioni dal JSON...")
sezioni_testuali,titoli = carica_sezioni_da_json(JSON_PATH)

print("Caricamento embedder...")
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

print("Creazione indice FAISS...")
index = crea_e_salva_indice(sezioni_testuali, embedder, EMBEDDINGS_SAVE_PATH, METADATA_SAVE_PATH)

print("Avvio interfaccia Gradio...")
gr.ChatInterface(fn=answer_question_with_fallback_llm, title="Chatbot Agricolus").launch()

embeddings = embedder.embed_documents(sezioni_testuali)

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

# Assumiamo che:
# - embedding_array sia un array NumPy di shape (n_sezioni, embedding_dim)
# - titoli sia una lista di stringhe (uno per ogni embedding)

# Calcolo delle distanze e linkage gerarchico
linked = linkage(embeddings, method='ward')  # Puoi anche usare 'average' o 'complete'

# Plot del dendrogramma
plt.figure(figsize=(16, 8))
dendrogram(
    linked,
    orientation='bottom',
    labels=[t[:40] for t in titoli],  # Mostra solo i primi 40 caratteri dei titoli
    distance_sort='descending',
    show_leaf_counts=True,
    leaf_rotation=90
)
plt.title("Dendrogramma delle sezioni del manuale")
plt.tight_layout()
plt.show()


query_embedding = embedder.embed_query(question)



import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

# embeddings delle sezioni (già generati)
# titoli = lista delle etichette
# embedder = già inizializzato

# Calcolo embedding della domanda
question = " Che genere di file posso caricare per inserire una mappa di vigore?"
domanda = " Che genere di file posso caricare per inserire una mappa di vigore?"
embedding_domanda = embedder.embed_query(domanda)

# Stacka gli embeddings delle sezioni con quello della domanda
all_embeddings = np.vstack([embedding_array, embedding_domanda])
all_labels = titoli + ["[DOMANDA]"]

# Riduzione dimensionale (t-SNE)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_embeddings = tsne.fit_transform(all_embeddings)

# Crea DataFrame per Plotly
import pandas as pd
df = pd.DataFrame(tsne_embeddings, columns=["x", "y"])
df["titolo"] = all_labels
df["tipo"] = ["sezione"] * len(titoli) + ["domanda"]

# Plot interattivo
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="tipo",
    hover_data=["titolo"],
    title="Distribuzione embeddings + domanda (t-SNE)"
)
fig.update_traces(marker=dict(size=8))
fig.show()