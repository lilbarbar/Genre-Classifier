import os
import random
import string
import math



# country_dir = '/Users/barilebari/Desktop/MyProjects/Genre-Classifier/kNN-data/country'
# country_songs = []
pop_dir = '/Users/barilebari/Desktop/MyProjects/Genre-Classifier/kNN-data/pop'
pop_songs = []
gospel_dir = '/Users/barilebari/Desktop/MyProjects/Genre-Classifier/kNN-data/gospel'
gospel_songs = []
rap_dir = '/Users/barilebari/Desktop/MyProjects/Genre-Classifier/kNN-data/rap'
rap_songs = []

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



class Song:


    def __init__(self, lyrics):
        self.lyrics = lyrics

        self.lyrics = self.lyrics.translate(str.maketrans('', '', string.punctuation)).upper()
        self.lyric_list = self.lyrics.split()
        self.vector = self.get_vector()
        self.magnitude = self.get_magnitude()
        self.unit_vector = self.get_unit_vector()


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
    

def normalize(vec):
    mag_squared = 0
    for i in vec:
        mag_squared += vec[i]**2
    
    if mag_squared == 0:
        mag_squared = 1  # avoid division by zero

    
    for i in vec:
        vec[i] = vec[i] / (mag_squared**.5)
    
    return vec




def generate_avg_dict(dicts):
    out = {}
    for smalldict in dicts:
        for j in smalldict:
            if j not in out:
                out[j] = 0
            out[j] += smalldict[j]
    
    for i in out:
        out[i] = out[i]/len(dicts)
    
    return normalize(out)

def generate_avg_dict_normal(dicts):
    out = {}
    for smalldict in dicts:
        for j in smalldict:
            if j not in out:
                out[j] = 0
            out[j] += smalldict[j]
    
    for i in out:
        out[i] = out[i]/len(dicts)
    
    return normalize(out)

def dictionary_dot_product(d1, d2):
    out = 0
    for i in d1:
        if i in d2:
            out += d1[i] * d2[i]
        
    return out


if __name__ == '__main__':
    overall_dict_words = set()

    # for filename in os.listdir(country_dir):
    #     if filename.endswith(".txt"):  # or any file extension
    #         file_path = os.path.join(country_dir, filename)
    #         with open(file_path, "r", encoding="utf-8") as f:
    #             content = f.read()
    #             country_songs.append(Song(content))
    

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
    
    



    input('Press Enter to vectorize the songs by lyric freqency and to use k-nearest neighbors to predict the genre of the song: ')

    print('\f'*100)
    allsongs = rap_songs + gospel_songs + pop_songs
    # set_country_songs = set(country_songs)
    set_rap_songs = set(rap_songs)
    set_gospel_songs = set(gospel_songs)
    set_pop_songs = set(pop_songs)

    new_contents = None
    with open('newinput.txt', 'r') as f:
        new_contents = f.read()  # reads the whole file as a string
    


    new_song = Song(new_contents)


    k = 3

    closest_songs = []

    for song in allsongs:
        if len(closest_songs) < 2*k + 1:
            closest_songs.append((song, new_song.dot_product(song)))                
        else:
            similar_score = new_song.dot_product(song)
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
    centroid1 = {}
    centroid2 = {}
    centroid3 = {}
    for i in overall_dict_words:
        centroid1[i] = random.randint(0, 20)
        centroid2[i] = random.randint(0, 20)
        centroid3[i] = random.randint(0, 20)
    centroid1 = normalize(centroid1)
    centroid2 = normalize(centroid2)
    centroid3 = normalize(centroid3)

    

    group1 = set()
    group2 = set()
    group3 = set()
    newgroup1 = set()
    newgroup2 = set()
    newgroup3 = set()
    
    for i in allsongs:

        similar_scores = [(dictionary_dot_product(centroid1, i.unit_vector), newgroup1),  (dictionary_dot_product(centroid2, i.unit_vector), newgroup2), (dictionary_dot_product(centroid3, i.unit_vector), newgroup3)]
        similar_scores.sort(key =lambda x:x[0])
        similar_scores[-1][1].add(i)



    g1vectors = [i.unit_vector for i in newgroup1]
    g2vectors = [i.unit_vector for i in newgroup2]
    g3vectors = [i.unit_vector for i in newgroup3]
    centroid1 = generate_avg_dict(g1vectors)
    centroid2 = generate_avg_dict(g2vectors)
    centroid3 = generate_avg_dict(g3vectors)

    iterations = 0
    while iterations < 1000 and ((group1 != newgroup1 and group2 != newgroup2) and (group3 != newgroup3)):
        group1 = newgroup1
        group2 = newgroup2
        group3 = newgroup3
        newgroup1 = set()
        newgroup2 = set()
        newgroup3 = set()
        for i in allsongs:
            similar_scores = [(dictionary_dot_product(centroid1, i.unit_vector), newgroup1),  (dictionary_dot_product(centroid2, i.unit_vector), newgroup2), (dictionary_dot_product(centroid3, i.unit_vector), newgroup3)]
            similar_scores.sort()
            similar_scores[-1][1].add(i)

        g1vectors = [i.unit_vector for i in newgroup1]
        g2vectors = [i.unit_vector for i in newgroup2]
        g3vectors = [i.unit_vector for i in newgroup3]
        centroid1 = generate_avg_dict(g1vectors)
        centroid2 = generate_avg_dict(g2vectors)
        centroid3 = generate_avg_dict(g3vectors)
        
    print(iterations)
    x1 = (dictionary_dot_product(centroid1, new_song.unit_vector), 'group1', centroid1)
    x2 = (dictionary_dot_product(centroid2, new_song.unit_vector), 'group2', centroid2)
    x3 = (dictionary_dot_product(centroid3, new_song.unit_vector), 'group3', centroid3)

    dot_products_for_song = [x1, x2, x3]

    dot_products_for_song.sort(key = lambda x: x[0])
    print("Using K means clustering, the song is most likely in " + dot_products_for_song[-1][1])
    words_in_centroid = []
    for i in dot_products_for_song[-1][2]:
        tup = (i, dot_products_for_song[-1][2][i])
        words_in_centroid.append(tup)
    
    words_in_centroid.sort(key=lambda x: x[1], reverse=True)
    print('Words in common and their freq')
    
    ix = 0
    for i in words_in_centroid:
        if i[0] not in stopwords_nlp_200:
            print(i)
            ix +=1
        if ix > 10:
            break
    
    print()
    print()
    input('Press enter to predict the genre of the song using premade clusters for k-means clustering: ')
    print('\f'*100)
    pop_avg = generate_avg_dict_normal([i.unit_vector for i in pop_songs])
    gospel_avg = generate_avg_dict_normal([i.unit_vector for i in gospel_songs])
    rap_avg = generate_avg_dict_normal([i.unit_vector for i in rap_songs])

    pop_score = dictionary_dot_product(pop_avg, new_song.unit_vector)
    gospel_score = dictionary_dot_product(gospel_avg, new_song.unit_vector)
    rap_score = dictionary_dot_product(rap_avg, new_song.unit_vector)


    pop_sig_score = (math.e ** pop_score)/ ((math.e**pop_score)+ (math.e**gospel_score) + (math.e**rap_score))
    gospel_sig_score = (math.e ** gospel_score)/ ((math.e**pop_score)+ (math.e**gospel_score) + (math.e**rap_score))

    rap_sig_score = (math.e ** rap_score)/ ((math.e**pop_score)+ (math.e**gospel_score) + (math.e**rap_score))

    print('Pop percentage', pop_sig_score * 100)
    print('Gospel percentage', gospel_sig_score * 100)
    print('Rap percentage', rap_sig_score * 100)



    

    






        
        





        
























