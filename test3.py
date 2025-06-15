from gensim.models import Word2Vec, FastText
import os
import random
import string
import numpy as np
import math


# Example corpus: list of tokenized sentences
sentences = [
    ["the", "quick", "brown", "fox"],
    ["the", "lazy", "dog"],
    ["the", "fox", "jumps", "over", "the", "dog"],
    ["hello", "world"]
]

stopwords_nlp_200 = {
    'A', 'ABOUT', 'ABOVE', 'AFTER', 'AGAIN', 'AGAINST', 'ALL', 'AM', 'AN', 'AND',
    'ANY', 'ARE', 'AREN\'T', 'AS', 'AT', 'BE', 'BEEN', 'BEFORE', 'BEING', 'BELOW',
    'BETWEEN', 'BOTH', 'BUT', 'BY', 'CAN', 'CAN\'T', 'CANNOT', 'COULD', 'COULDN\'T',
    'DID', 'DIDN\'T', 'DO', 'DOES', 'DOESN\'T', 'DOING', 'DON\'T', 'DOWN', 'DURING',
    'EACH', 'FEW', 'FOR', 'FROM', 'FURTHER', 'HAD', 'HADN\'T', 'HAS', 'HASN\'T',
    'HAVE', 'HAVEN\'T', 'HAVING', 'HE', 'HE\'D', 'HE\'LL', 'HE\'S', 'HER', 'HERE',
    'HERE\'S', 'HERS', 'HERSELF', 'HIM', 'HIMSELF', 'HIS', 'HOW', 'HOW\'S', 'I',
    'I\'D', 'I\'LL', 'I\'M', 'I\'VE', 'IF', 'IN', 'INTO', 'IS', 'ISN\'T', 'IT',
    'IT\'S', 'ITS', 'ITSELF', 'LET\'S', 'ME', 'MORE', 'MOST', 'MUST', 'MUSTN\'T',
    'MY', 'MYSELF', 'NO', 'NOR', 'NOT', 'OF', 'OFF', 'ON', 'ONCE', 'ONLY',
    'OR', 'OTHER', 'OUGHT', 'OUR', 'OURS', 'OURSELVES', 'OUT', 'OVER', 'OWN', 'SAME',
    'SHALL', 'SHAN\'T', 'SHE', 'SHE\'D', 'SHE\'LL', 'SHE\'S', 'SHOULD', 'SHOULDN\'T',
    'SO', 'SOME', 'SUCH', 'THAN', 'THAT', 'THAT\'S', 'THE', 'THEIR', 'THEIRS', 'THEM',
    'THEMSELVES', 'THEN', 'THERE', 'THERE\'S', 'THESE', 'THEY', 'THEY\'D', 'THEY\'LL',
    'THEY\'RE', 'THEY\'VE', 'THIS', 'THOSE', 'THROUGH', 'TO', 'TOO', 'UNDER', 'UNTIL', 'UP',
    'VERY', 'WAS', 'WASN\'T', 'WE', 'WE\'D', 'WE\'LL', 'WE\'RE', 'WE\'VE', 'WERE',
    'WEREN\'T', 'WHAT', 'WHAT\'S', 'WHEN', 'WHEN\'S', 'WHERE', 'WHERE\'S', 'WHICH',
    'WHILE', 'WHO', 'WHO\'S', 'WHOM', 'WHY', 'WHY\'S', 'WILL', 'WITH', 'WITHIN',
    'WITHOUT', 'WON\'T', 'WOULD', 'WOULDN\'T', 'YES', 'YET', 'YOU', 'YOU\'D', 'YOU\'LL',
    'YOU\'RE', 'YOU\'VE', 'YOUR', 'YOURS', 'YOURSELF', 'YOURSELVES', 'EVERY', 'EVERYONE',
    'EVERYTHING', 'EVERYBODY', 'ANYBODY', 'ANYONE', 'ANYTHING', 'NOBODY', 'NOTHING',
    'NOWHERE', 'MINE', 'FEWER', 'ENOUGH', 'SOMEONE', 'SOMEBODY', 'SEVERAL', 'MANY',
    'WHOEVER', 'WHOMEVER', 'WHEREVER', 'WHENEVER', 'WHICHEVER', 'WHATEVER', 'LEAST',
    'BESIDE', 'BESIDES', 'TOWARD', 'TOWARDS', 'UPON', 'AMONG', 'ALTHOUGH', 'UNLESS',
    'THOUGH', 'EVEN', 'RATHER', 'INDEED', 'NEITHER', 'EITHER', 'MAY', 'MIGHT',
    'SHOULD\'VE', 'COULDN\'T\'VE', 'WOULDN\'T\'VE', 'MUST\'VE', 'MIGHT\'VE', 'SHAN\'T',
    'HAVING', 'SOMEWHAT', 'HITHERTO', 'THEREIN', 'THEREOF', 'WHEREUPON', 'WHEREIN'
}

# sentences = """ 

# In the heart of an ancient forest, a peculiar strain of coffee beans known as ChronoBeans grew beneath the gnarled branches of time-worn trees. Legend whispered that these beans absorbed the energy of passing centuries, gaining the ability to warp the very essence of time itself. When brewed, the coffee didn’t just give a jolt of caffeine—it gave sips of moments past and glimpses of the future. One day, a curious barista stumbled upon these beans and roasted them with care, only to find herself standing in a bustling medieval marketplace, surrounded by knights and merchants. With each sip, the world around her flickered between eras, a kaleidoscope of history’s greatest hits. The ChronoBeans had unlocked a secret long forgotten: the power of time-traveling espresso.



# """
# sentences = sentences.split()

# # Train the Word2Vec model
# model = FastText(
#     [sentences],        # your corpus (list of tokenized sentences)
#     vector_size=100,  # dimensionality of word vectors
#     window=5,         # context window size
#     min_count=1,      # ignore words with freq < 1
#     workers=4,        # number of threads
#     sg=1              # 1 for skip-gram, 0 for CBOW
# )

# # Save the model
# model.save("my_word2vec.model")

# # Access the vector for a word

# vector = model.wv["ball"]
# # print(vector)

# similar_words = model.wv.most_similar("coffee")
# print(similar_words)





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
        self.lyric_list = self.lyrics.split()

        self.vector = self.get_vector()
        self.magnitude = self.get_magnitude()
        self.unit_vector = self.get_unit_vector()



    def get_fast_text_lyric_vectors (self):
        out = []
        model = FastText.load('my_fast_text.model')
        for i in self.lyrics:
            out.append(model.wv[i])
        return out

    def get_overall_fast_text_vector (self):

        # vecs = self.get_fast_text_lyric_vectors()          # list[ndarray]
        # if not vecs:                                       # all tokens OOV
        #     return np.zeros(self.model.vector_size, dtype=np.float32)

        total =  (sum(vec for vec in self.get_fast_text_lyric_vectors()))
        return total/np.linalg.norm(total)

    def dot_product_fast_text (self, other):
        self_v = self.get_overall_fast_text_vector()
        other_v = other.get_overall_fast_text_vector()
        # out = 0
        # for i in self_v:
        #     if i in other_v:
        #         out += self_v[i] * other_v[i]
        return np.dot(self_v, other_v)



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
    return float(np.dot(v1, v2))

def generate_avg_vector(vecs):
    if len(vecs) == 0:
        return np.zeros(vec_size)
    out = sum(vecs)/len(vecs)
    return out / np.linalg.norm(out)


    
    # return normalize(out)

if __name__ == '__main__':
    input('Press Enter to vectorize the songs using FastText deep vectors and to use k-nearest neighbors to predict the genre of the song: ')
    print('\f'*100)

    overall_dict_words = set()
    fast_text_songs = []
    for filename in os.listdir(country_dir):
        if filename.endswith(".txt"):  # or any file extension
            file_path = os.path.join(country_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                content_list = content.translate(str.maketrans('', '', string.punctuation)).split()
                fast_text_songs.append(content_list)
                country_songs.append(Song(content))
    

    for filename in os.listdir(pop_dir):
        if filename.endswith(".txt"):  # or any file extension
            file_path = os.path.join(pop_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                content_list = content.translate(str.maketrans('', '', string.punctuation)).split()
                fast_text_songs.append(content_list)
                pop_songs.append(Song(content))
    

    for filename in os.listdir(gospel_dir):
        if filename.endswith(".txt"):  # or any file extension
            file_path = os.path.join(gospel_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                content_list = content.translate(str.maketrans('', '', string.punctuation)).split()
                fast_text_songs.append(content_list)
                gospel_songs.append(Song(content))
    

    for filename in os.listdir(rap_dir):
        if filename.endswith(".txt"):  # or any file extension
            file_path = os.path.join(rap_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                content_list = content.translate(str.maketrans('', '', string.punctuation)).split()
                fast_text_songs.append(content_list)
                rap_songs.append(Song(content))
    
    allsongs =  rap_songs + gospel_songs + pop_songs

    sentences = []
    for i in allsongs:
        sentences.append(i.lyric_list)
    vec_size = 20
    model = FastText(
    sentences,        # your corpus (list of tokenized sentences)
    vector_size=vec_size,  # dimensionality of word vectors
    window=5,         # context window size
    min_count=1,      # ignore words with freq < 1
    workers=4,        # number of threads
    sg=1              # 1 for skip-gram, 0 for CBOW
    )

# Save the model
    model.save("my_fast_text.model")

    # Access the vector for a word

    vector = model.wv["ball"]
    # print(vector)

    similar_words = model.wv.most_similar("coffee")
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
            closest_songs.append((song, new_song.dot_product_fast_text(song)))                
        else:
            similar_score = new_song.dot_product_fast_text(song)
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

    input('''Press Enter to vectorize the songs by lyric freqency and to use k-means cluster to categorize a group of songs and get the most freqent words in said group: ''')
    print('\f'*100)

        
    for song in allsongs:
            for word in song.lyric_list:
                overall_dict_words.add(word)
    centroid1 = np.random.rand(vec_size)
    centroid2 = np.random.rand(vec_size)
    centroid3 = np.random.rand(vec_size)
    centroid4 = np.random.rand(vec_size)
    # for i in overall_dict_words:
    #     centroid1[i] = random.randint(0, 20)
    #     centroid2[i] = random.randint(0, 20)
    #     centroid3[i] = random.randint(0, 20)
    #     centroid4[i] = random.randint(0, 20)
    centroid1 = (centroid1)/(np.linalg.norm(centroid1))
    centroid2 = (centroid2)/(np.linalg.norm(centroid2))
    centroid3 = (centroid3)/(np.linalg.norm(centroid3))
    centroid4 = (centroid4)/(np.linalg.norm(centroid4))

    

    group1 = set()
    group2 = set()
    group3 = set()
    group4 = set()
    newgroup1 = set()
    newgroup2 = set()
    newgroup3 = set()
    newgroup4 = set()
    
    for i in allsongs:
        similar_scores = [(np_dot_product(centroid1, i.get_overall_fast_text_vector()), newgroup1), (np_dot_product(centroid2, i.get_overall_fast_text_vector()), newgroup2), (np_dot_product(centroid3, i.get_overall_fast_text_vector()), newgroup3), (np_dot_product(centroid4, i.get_overall_fast_text_vector()), newgroup4)]
        similar_scores.sort(key =lambda x:x[0])
        similar_scores[-1][1].add(i)



    g1vectors = [i.get_overall_fast_text_vector() for i in newgroup1]
    g2vectors = [i.get_overall_fast_text_vector() for i in newgroup2]
    g3vectors = [i.get_overall_fast_text_vector() for i in newgroup3]
    g4vectors = [i.get_overall_fast_text_vector() for i in newgroup4]
    centroid1 = generate_avg_vector(g1vectors)
    centroid2 = generate_avg_vector(g2vectors)
    centroid3 = generate_avg_vector(g3vectors)
    centroid4 = generate_avg_vector(g4vectors)

    iterations = 0
    while iterations < 1000 and ((group1 != newgroup1 and group2 != newgroup2) and (group3 != newgroup3 and group4 != newgroup4)):
        group1 = newgroup1
        group2 = newgroup2
        group3 = newgroup3
        group4 = newgroup4
        newgroup1 = set()
        newgroup2 = set()
        newgroup3 = set()
        newgroup4 = set()
        for i in allsongs:
            similar_scores = [(np_dot_product(centroid1, i.get_overall_fast_text_vector()), newgroup1), (np_dot_product(centroid2, i.get_overall_fast_text_vector()), newgroup2), (np_dot_product(centroid3, i.get_overall_fast_text_vector()), newgroup3), (np_dot_product(centroid4, i.get_overall_fast_text_vector()), newgroup4)]
            similar_scores.sort(key =lambda x:x[0])
            similar_scores[-1][1].add(i)

        g1vectors = [i.get_overall_fast_text_vector() for i in newgroup1]
        g2vectors = [i.get_overall_fast_text_vector() for i in newgroup2]
        g3vectors = [i.get_overall_fast_text_vector() for i in newgroup3]
        g4vectors = [i.get_overall_fast_text_vector() for i in newgroup4]
        centroid1 = generate_avg_vector(g1vectors)
        centroid2 = generate_avg_vector(g2vectors)
        centroid3 = generate_avg_vector(g3vectors)
        centroid4 = generate_avg_vector(g4vectors)
        
    print(iterations)
    x1 = (np_dot_product(centroid1, new_song.get_overall_fast_text_vector() ), 'group1', centroid1)
    x2 = (np_dot_product(centroid2, new_song.get_overall_fast_text_vector() ), 'group2', centroid2)
    x3 = (np_dot_product(centroid3, new_song.get_overall_fast_text_vector() ), 'group3', centroid3)
    x4 = (np_dot_product(centroid4, new_song.get_overall_fast_text_vector() ), 'group4', centroid4)

    dot_products_for_song = [x1, x2, x3, x4]

    dot_products_for_song.sort(key = lambda x: x[0])
    print("Using K means clustering, the song is most likely in " + dot_products_for_song[-1][1])
    # words_in_centroid = []
    # for i in dot_products_for_song[-1][2]:
    #     tup = (i, dot_products_for_song[-1][2][i])
    #     words_in_centroid.append(tup)
    
    # words_in_centroid.sort(key=lambda x: x[1], reverse=True)
    print('Words in common and their freq')
    similar_words = model.wv.similar_by_vector(dot_products_for_song[-1][2], topn=20)

    print(similar_words)    
    # ix = 0
    # for i in words_in_centroid:
    #     if i[0] not in stopwords_nlp_200:
    #         print(i)
    #         ix +=1
    #     if ix > 10:
    #         break
    


    input('Press enter to predict the genre of the song using premade clusters for k-means clustering: ')
    print('\f'*100)

    pop_avg = generate_avg_vector([i.get_overall_fast_text_vector() for i in pop_songs])
    gospel_avg = generate_avg_vector([i.get_overall_fast_text_vector() for i in gospel_songs])
    rap_avg = generate_avg_vector([i.get_overall_fast_text_vector() for i in rap_songs])

    pop_score = np_dot_product(pop_avg, new_song.get_overall_fast_text_vector())
    gospel_score = np_dot_product(gospel_avg, new_song.get_overall_fast_text_vector())
    rap_score = np_dot_product(rap_avg, new_song.get_overall_fast_text_vector())


    pop_sig_score = (math.e ** pop_score)/ ((math.e**pop_score)+ (math.e**gospel_score) + (math.e**rap_score))
    gospel_sig_score = (math.e ** gospel_score)/ ((math.e**pop_score)+ (math.e**gospel_score) + (math.e**rap_score))

    rap_sig_score = (math.e ** rap_score)/ ((math.e**pop_score)+ (math.e**gospel_score) + (math.e**rap_score))

    print('Pop percentage', pop_sig_score * 100)
    print('Gospel percentage', gospel_sig_score * 100)
    print('Rap percentage', rap_sig_score * 100)

