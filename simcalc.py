import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

vec1 = np.random.rand(384)
vec2 = np.random.rand(384)
# print(f'vector 1 is {vec1} and vector 2  is {vec2}')
print("*******")
print("*******")
print("*******")
print("*******")
print("*******")

if __name__ == "__main__":
    vec1_2d = vec1.reshape(1, -1)
    vec2_2d = vec2.reshape(1, -1)
    # print(f'vector 1 2D is {vec1_2d} and vector 2 2D is {vec2_2d}')

    cos_sim = cosine_similarity(vec1_2d, vec2_2d)
    print("cosine similarity: ", cos_sim[0][0])
    