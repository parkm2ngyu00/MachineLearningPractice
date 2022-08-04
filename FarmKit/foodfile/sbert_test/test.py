# 기본 파일 읽고 배열에 저장하기
f = open('food_name.csv', encoding='utf-8')
food_names = f.readlines()


f = open('food_explain.csv', encoding='utf-8')
food_descs = f.readlines()

f = open('food_ingre.csv', encoding='utf-8')
food_ingres = f.readlines()

# 파일 라인수 같은지 보기
print(len(food_descs))
print(len(food_ingres))
print(len(food_names))

# 설명 + 재료의 문장 생성
food_ingres_descs = []
for i in range(len(food_ingres)):
    ingres_descs = "The description of the food as follows. " + food_descs[i] + " And, the ingredients of food includes " + food_ingres[i]
    print(ingres_descs)
    food_ingres_descs.append(ingres_descs)
    
    
# Sentence Bert 
from sentence_transformers import SentenceTransformer, util
import torch
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-mpnet-base-v2')


# 재료만 embedding
print('food_ingres embedding started ... ')
ingres_embeddings = model.encode(food_ingres)

# Description 만 embedding
print('food_descs embedding started ... ')
descs_embeddings = model.encode(food_descs)

# 설명 + 재료의 embedding
print('food_ingres_descs embedding started ... ')
ingres_descs_embeddings = model.encode(food_ingres_descs)

 # 유사도 계산해보기 ... 
ingres_scores_dist = torch.cdist(torch.tensor(ingres_embeddings), torch.tensor(ingres_embeddings))
descs_scores_dist = torch.cdist(torch.tensor(descs_embeddings), torch.tensor(descs_embeddings))
ingres_descs_scores_dist = torch.cdist(torch.tensor(ingres_descs_embeddings), torch.tensor(ingres_descs_embeddings))

ingres_scores = util.cos_sim(ingres_embeddings, ingres_embeddings)
descs_scores = util.cos_sim(descs_embeddings, descs_embeddings)
ingres_descs_scores = util.cos_sim(ingres_descs_embeddings, ingres_descs_embeddings)

def get_top(idx, scores, sortReverse) :
    pair = []
    for i in range(len(scores)):
        pair.append({'index':i, 'score':scores[idx][i]})
    pair = sorted(pair, key=lambda x: x['score'], reverse=sortReverse)
    ret = []
    for i in range(20):
        ret.append(food_names[pair[i]['index']])
    return ret

while (True):
    print('input food index: 0 ~ 2380')
    idx = int(input())
    print(food_names[idx])
    tops = get_top(idx, ingres_scores, True)
    print('ingredients top 20, cos sim')
    print(tops)
    print('\n\n')
    
    tops = get_top(idx, ingres_scores_dist, False)
    print('ingredients top 20, euclidian distance')
    print(tops)
    print('\n\n')
    
    tops = get_top(idx, descs_scores, True)
    print('description top 20, cos sim')
    print(tops)
    print('\n\n')
    
    
    tops = get_top(idx, descs_scores_dist, False)
    print('description top 20, euclidian distance')
    print(tops)
    print('\n\n')
    
    
    tops = get_top(idx, ingres_descs_scores, True)
    print('ingredients + description top 20, cos sim')
    print(tops)
    print('\n\n')
    
    
    tops = get_top(idx, ingres_descs_scores_dist, False)
    print('ingredients + description 20, euclidian distance')
    print(tops)
    print('\n\n')
    
