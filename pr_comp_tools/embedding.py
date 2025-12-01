from sentence_transformers import SentenceTransformer, util

def load_model():
    return SentenceTransformer('/商品PR生成LLMコンペ_埋め込みモデル/model')


def calc_simil(genSentenceLst, goalSentenceLst):
    if len(genSentenceLst) != len(goalSentenceLst):
        print("Length Failure")
        return
    embed_model=load_mode()
    maxSimil=[0.0, 0]
    minSimil=[0.0, 0]
    totalSimil=0.0
    for i in range(len(genSentenceLst)):
        emb1 = embed_model.encode(genSentenceLst[i], convert_to_tensor=True)
        emb2 = embed_model.encode(goalSentenceLst[i], convert_to_tensor=True)
        simil = util.cos_sim(emb1, emb2).item()
        if simil > maxSimil[0]:
            maxSimil[0]=simil
            maxSimil[1]=i
        if simil < min_simil[0]:
            minSimil[0]=simil
            minSimil[1]=i
        totalSimil+=simil
    print(f"AVE SIMILARITY : {totalSImil/len(genSentenceLst)}")
    print(f"MAX SIMILARITY : {maxSimil[0]}")
    print(f"MAX SIMILARITY GENERATED SENTENCE : {genSentenceLst[maxSimil[1]]}")
    print(f"MAX SIMILARITY TARGET SENTENCE : {goalSentenceLst[maxSimil[1]]}")
    print(f"MIN SIMILARITY GENERATED SENTENCE : {genSentenceLst[minSimil[1]]}")
    print(f"MIN SIMILARITY TARGET SENTENCE : {goalSentenceLst[minSimil[1]]}")        

if __name__ == "__main__":
    genSentenceLst = ["これはラーメンです", "これは醤油ラーメンです", "これはおいしい味噌ラーメンです"]
    goalSentenceLst = ["これは味噌ラーメンです","これは味噌ラーメンです","これは味噌ラーメンです"]
    calc_simil(genSentenceLst, goalSentenceLst)
