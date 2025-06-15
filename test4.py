from transformers import AutoTokenizer, AutoModel
import torch
import os
import random
import string
import numpy as np
import math
from gensim.models import Word2Vec, FastText





 # shape: (1, 768)
# sentence_embedding = embedding.mean(dim=1)  # shape: (1, 768)
# print(sentence_embedding) 


# def np_dot_product(v1, v2):
#     return np.dot(v1, v2)

# def generate_avg_vector(vecs):
#     if len(vecs) == 0:
#         return np.zeros(vec_size)
#     out = sum(vecs)/len(vecs)
#     return out / np.linalg.norm(out)

















country_dir = '/Users/barilebari/Desktop/MyProjects/Genre-Classifier/kNN-data/country'
country_songs = []
pop_dir = '/Users/barilebari/Desktop/MyProjects/Genre-Classifier/kNN-data/pop'
pop_songs = []
gospel_dir = '/Users/barilebari/Desktop/MyProjects/Genre-Classifier/kNN-data/gospel'
gospel_songs = []
rap_dir = '/Users/barilebari/Desktop/MyProjects/Genre-Classifier/kNN-data/rap'
rap_songs = []


class Song:


    def __init__(self, lyrics):
        self.lyrics = lyrics
        self.lyrics = self.lyrics.translate(str.maketrans('', '', string.punctuation)).upper()
        self.pure_lyrics = lyrics
        self.lyric_list = self.lyrics.split()
        self.vector = self.get_vector()
        self.magnitude = self.get_magnitude()
        self.unit_vector = self.get_unit_vector()
        self.song_vector = self.get_song_vector()



    def get_song_vector (self):
        inputs = tokenizer(self.lyrics, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state  # Contextual embeddings for each token
        x = embedding[:, 0, :].detach().numpy().flatten()
        return x / np.linalg.norm(x)



    def dot_song_vector(self, other):
        return np.dot(self.get_song_vector(), other.get_song_vector())


    def dot_product (self, other):
        out = 0
        for i in self.unit_vector:
            if i in other.unit_vector:
                out += self.unit_vector[i] * other.unit_vector[i]
        
        return out

    def get_magnitude(self):
        total_sum = 0
        for i in self.vector:
            total_sum += (self.vector[i] ** 2)
        
        return total_sum ** (.5)


    def get_unit_vector (self):
        out = {}
        mag = self.magnitude
        for i in self.vector:
            out[i] = (self.vector[i])/mag
        
        return out
        

    def get_vector (self):
        out = {}
        for i in self.lyric_list:
            if i not in out:
                out[i] = 0
            out[i] +=1

        return out


def np_dot_product(v1, v2):
    return np.dot(v1, v2)

def generate_avg_vector(vecs):
    if len(vecs) == 0:
        return np.zeros(512)
    out = sum(vecs)/len(vecs)
    return out / np.linalg.norm(out)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    input('Press Enter to vectorize the songs using FastText deep vectors and to use k-nearest neighbors to predict the genre of the song: ')
    print('\f'*100)
    overall_dict_words = set()
    fast_text_songs = []

    for filename in os.listdir(country_dir):
        if filename.endswith(".txt"):  # or any file extension
            file_path = os.path.join(country_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read() 
                country_songs.append(Song(content))
    

    for filename in os.listdir(pop_dir):
        if filename.endswith(".txt"):  # or any file extension
            file_path = os.path.join(pop_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                pop_songs.append(Song(content))
    

    for filename in os.listdir(gospel_dir):
        if filename.endswith(".txt"):  # or any file extension
            file_path = os.path.join(gospel_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                gospel_songs.append(Song(content))
    

    for filename in os.listdir(rap_dir):
        if filename.endswith(".txt"):  # or any file extension
            file_path = os.path.join(rap_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                rap_songs.append(Song(content))
    
    # all_songs = country_songs + gospel_songs + pop_songs + rap_songs

    allsongs =  rap_songs + gospel_songs + pop_songs

    # sentences = []
    # for i in allsongs:
    #     sentences.append(i.lyric_list)
    # vec_size = 20
    # model = FastText(
    # sentences,        # your corpus (list of tokenized sentences)
    # vector_size=vec_size,  # dimensionality of word vectors
    # window=5,         # context window size
    # min_count=1,      # ignore words with freq < 1
    # workers=4,        # number of threads
    # sg=1              # 1 for skip-gram, 0 for CBOW
    # )

# Save the model
    # model.save("my_fast_text.model")

    # Access the vector for a word

    # vector = model.wv["ball"]
    # # print(vector)

    # similar_words = model.wv.most_similar("coffee")
    # print(similar_words)


    
    # set_country_songs = set(country_songs)
    set_rap_songs = set(rap_songs)
    set_gospel_songs = set(gospel_songs)
    set_pop_songs = set(pop_songs)

    new_contents = None
    with open('newinput.txt', 'r') as f:
        new_contents = f.read()  # reads the whole file as a string
    


    new_song = Song(new_contents)
    # new_song.initialize_tf_idf_vectors(overall_dictionary)

    k = 3

    closest_songs = []
    for song in allsongs:
        if len(closest_songs) < 2*k + 1:
            closest_songs.append((song, new_song.dot_song_vector(song)))                
        else:
            similar_score = new_song.dot_song_vector(song)
            if similar_score>closest_songs[-1][1]:
                closest_songs.pop(-1)
                closest_songs.append((song, similar_score))
        closest_songs.sort(key=lambda x: x[1], reverse=True)
    

    pop_count = 0
    rap_count = 0
    gospel_count = 0
    # country_count = 0
    for i in closest_songs[:k]:
        if i[0] in set_pop_songs:
            pop_count+=1
        elif i[0] in set_rap_songs:
            rap_count +=1
        elif i[0] in set_gospel_songs:
            gospel_count +=1
        # elif i[0] in set_country_songs:
        #     country_count +=1
    

    index_to_go = k
    
    most = max(pop_count, rap_count, gospel_count)
    most_list = []
    if rap_count == most:
        most_list.append('RAP')
    if gospel_count == most:
        most_list.append('GOSPEL')
    if pop_count == most:
        most_list.append('POP')
    # if country_count == most:
    #     most_list.append('COUNTRY')
   
    while len(most_list) != 1:
        next_song = closest_songs[index_to_go]
        if next_song[0] in set_pop_songs:
            pop_count+=1
        elif next_song[0] in set_rap_songs:
            rap_count +=1
        elif next_song[0] in set_gospel_songs:
            gospel_count +=1
        # elif next_song[0] in set_country_songs:
        #     country_count +=1
        most = max(pop_count, rap_count, gospel_count)
        most_list = []
        if rap_count == most:
            most_list.append('RAP')
        if gospel_count == most:
            most_list.append('GOSPEL')
        if pop_count == most:
            most_list.append('POP')
        # if country_count == most:
        #     most_list.append('COUNTRY')
        index_to_go +=1

        
    
    print('Using k nearest neighbors, the predicted genre of the song given is: ' + most_list[0][:10])


    input('Press enter to predict the genre of the song using premade clusters for k-means clustering: ')
    print('\f'*100)

    pop_avg = generate_avg_vector([i.song_vector for i in pop_songs])
    gospel_avg = generate_avg_vector([i.song_vector for i in gospel_songs])
    rap_avg = generate_avg_vector([i.song_vector for i in rap_songs])

    pop_score = np_dot_product(pop_avg, new_song.song_vector)
    gospel_score = np_dot_product(gospel_avg, new_song.song_vector)
    rap_score = np_dot_product(rap_avg, new_song.song_vector
                               )


    pop_sig_score = (math.e ** pop_score)/ ((math.e**pop_score)+ (math.e**gospel_score) + (math.e**rap_score))
    gospel_sig_score = (math.e ** gospel_score)/ ((math.e**pop_score)+ (math.e**gospel_score) + (math.e**rap_score))

    rap_sig_score = (math.e ** rap_score)/ ((math.e**pop_score)+ (math.e**gospel_score) + (math.e**rap_score))

    print('Pop percentage', pop_sig_score * 100)
    print('Gospel percentage', gospel_sig_score * 100)
    print('Rap percentage', rap_sig_score * 100)



