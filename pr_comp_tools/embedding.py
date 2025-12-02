from sentence_transformers import SentenceTransformer, util
import torch

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer('model',device = device)

def calc_simil(genSentenceLst, goalSentenceLst):
    embed_model=load_model()
    datas=[]
    for i in range(len(genSentenceLst)):
        emb1 = embed_model.encode(genSentenceLst[i], convert_to_tensor=True)
        emb2 = embed_model.encode(goalSentenceLst[i], convert_to_tensor=True)
        simil = util.cos_sim(emb1, emb2).item()
        datas.append({"similarity": similarity, "generated_pr": generated_pr, "correct_pr": correct_pr})
    return datas

if __name__ == "__main__":

    genSentenceLst = ["これはラーメンです", "これは醤油ラーメンです", "これはおいしい味噌ラーメンです"]
    goalSentenceLst = ["これは味噌ラーメンです","これは味噌ラーメンです","これは味噌ラーメンです"]
    calc_simil(genSentenceLst, goalSentenceLst)
