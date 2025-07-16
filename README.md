# NLP-for-Machine-learning
Text preprocessing >>> what we have learn ? 

sentiment Analysis : 

            
Dataset >>> text preprocessing >>>>>>>> Text preprocessing part 2 >>>>
            * Tokenization              *Stemming
            * Lower case the words      * Lemmatization
            * Regular Expertion         * Stopwords



>>>>>> (text ---> vectors)>>>>>>>>> ML Algorithms  
       * one hot encoded 
       * Bag of words (BOW)
       * Tf-IDF
       * word2vec
       * Avg word2vec


---------------------------------------------------------------------------------

(text ---> vectors) 

1) one hot Encoded 

D1> the food is good 
D2> the food is bad
D3> pizza is Amazing

                       
* Vocabulary {unique words} >>>>>

the food is good bad pizza Amazing

for D1> 

1    0    0  0    0   0     0  =  
0    1    0  0    0   0     0  = 
0    0    1  0    0   0     0 
0    0    0  1    0   0     0  


for D2 > 

1    0    0  0    0   0     0 
0    1    0  0    0   0     0 
0    0    1  0    0   0     0 
0    0    0  0    1   0     0 

------------------------------------------------------------


the Advantages and disAdvatages of (one hot encoded) :

* Advantages >>>  
1. Easy to implement with python 
   ( sklearn > one hotencoder , pd.get_dummies() ) 




* Disadvanages :

sparse matrics >> overfitting 

ml Algorithm >>>>> we need fixed size I/p 

no semantic meaning is basically getting 

out of vocabulary




-------------------------------------------------------------------------


Bag of words (BOW) >>>             low case all the world 

text                           o/p  
He is a good boy                1      >>>> s1 >> good boy 
she is a good girl              1      >>>> s2 >> good girl
Boy and girl are good           1      >>>> s3 >> boy girl good 



  vocabulary          frequency 
   good                 3 
  
   boy                  2

   girl                 2        

     good boy girl
s1   [1  1   0] 

s2  [1   0   1] 

s3  [1   1   1] 



Binary Bow  and Bow  >>>>>>>>
{1 , 0 }        { count will get update back on frequency } 




_________________________________________________________________________

BoW ......

Advantage :                           DisAdvantage 

1) simple and Intuitive                * sparce matrix or array >>> overfiting  
                                       * ordering of the world is getting change 
2) fixed sized I/p >> ml Algorithm     * out of vocabulary (oov) 
                                       * sementic meaning is still not captured

----------------------------------------------------------------------------------------------



*** (N-grams)     eg : b


s1: the food is good                                 vocabulary : food not good 
                                                                   1    0    1
s2: the food is not good                                           1     1   1 


   [ food  not  good     foodgood     foodnot       notgood ]

s1    1     0     1         1           0             0      

s2    1      1    1         0           1              1    



sklearn >>>>> n-grams = (1, 1 ) >>>> unigrams 
                      = (1 , 2) >>>> unigrams , bigrams 
                      = (1 , 3) >>>> unigrams , bigrams , trigrams 
                      = (2 , 3) >>>> bigrams , trigrams 



--------------------------------------------------------------------------------------------

### TF-IDF >>> [Term frequency and inverse document frequency ]


as we work in bag of words >>> u can go and see ...

s1 >> good boy 
s2 >> good girl 
s3 >> boy girl good 



------------------------------------------------

term frequency >> no of rep of words in sentence / no of words in sentence

IDF = loge(no of sentences / no og sentences containing the word 



term requency 
        s1       s2       s3 

good   1/2       1/2     1/3

boy    1/2      0       1/3

girl   0         1/2    1/3


*IDF 

words        IDF
good          log(3/3)=0

boy           log(3/2) 

girl          log(3/2) 


------------------------------------------------------------------------------


final TF-IDF    = term frequency * IDF


    good         boy                   girl 

s1   0            1/2*log(3/2)          0


s2  0              0                    1/2*log(3/2)


s3  0              1/3log(3/2)          1/3 log(3/2) 




------------------------------------------------------------------------


TF-IDF 


Advantages                        DisAdvantage 


1. Intuitive                       1. Sparsity is still exist .

2. Fixed Size >>> vocab size       2. oov (out of vocabulary) 

3. word importance is getting 
captured 


----------------------------------------------------------------

### Word Embeddengs 

in natural language processing (NLP), word embedding is a term used for the representation of words for text 
analysis , typically in the form of a rea_valued vector that encoded the meaning of the word such that the words that are 
closer in the vector space are expected to be similar in meaning . 




Word Embeddings >>>

1. count or frequency   >>> 1. OHE           2. BOW             3.Tf_IDF


2. Deep learning trained model >>> 1. word2vec >> a.CBOW(countinous Bag of words)   
                                                  b.skipgram  


--------------------------------------------------------------------------------------

# Word2vec: >>>> feature Represntation .

word2vec is a technique for natural language processing published in 2013. The word2vec algorithm uses a neural network model 
to learn word associations from a large corpus of text.once trained , such a model can detect 
synonymous words or suggest additional words for a partial sentence. As the name implies, word2vec represents each 
distinct word with a particular list of numbers called a vector . 


---------------------------------------------------------------------------

vocabulary >> unique words >>> corpus 


         Boy   Girl   King   Queen Apple  Mango 


Gender   -1    1      0.94    -0.93  0.004 0.005

Royel 

Age 

food 
.
.
.
..
.
300 dimensions feature
for every word 
--------------------------------------------------

Relationships between words are preserved.
Example:
vec("king") - vec("man") + vec("woman") ≈ vec("queen")



( Cosine Similarity ) >>> 

let say the angle between 2 vector is 45 


Distance = 1- cosine similarity 

Cosine sim = cos(45)  

so the distance = 1- 0.7071 = 0.3


*Cosine similarity:

is commonly used with Word2Vec to measure how similar two words are based on their vector representations.

Why Cosine Similarity?
Word2Vec turns each word into a vector of real numbers in a high-dimensional space (like 100 or 300 dimensions).
To find how similar two words are, we don’t just subtract them — instead, we look at the angle between their vectors.

That’s where cosine similarity comes in.



------------------------------------------------------------------------------------------------------------------



### Word2vec CBOW   


as we know >> 



Ann , loss , optimizers 

* Word2vec >>> *cbow                    (pretrained Model) (train A model from scratch ) 
               *skipgramm


----------------------------------

at first we will start with (CBOW) >>>[continous Bag of words]

What is Word2Vec (CBOW)?
Goal: Learn to represent each word as a vector (embedding) such that similar words have similar vectors.

CBOW predicts the target word given its context words.



CBOW Architecture (Intuition)
Example sentence:

plaintext
Copy
Edit
"The cat sits on the mat"
Let’s say our context window = 2

For the word "sits", the context is:

plaintext
Copy
Edit
["the", "cat", "on", "the"]
So training data =
X (input) = ["the", "cat", "on", "the"]
y (output) = "sits"

🧠 The model learns that if we see “the”, “cat”, “on”, “the” around — then “sits” is likely the center word.

🧠 CBOW Architecture (Layers):
Input layer: One-hot encoded context words

Hidden layer: Shared embedding layer (size = 100, for example)

Output layer: Softmax over vocabulary to predict target word

🔢 Example with Vocabulary
Let’s assume a small vocab of 5 words:
["king", "queen", "man", "woman", "child"]

Step 1: Context = ["king", "man"], Target = "queen"
Step 2: One-hot encode context:
ini
Copy
Edit
king  = [1, 0, 0, 0, 0]  
man   = [0, 0, 1, 0, 0]
Step 3: Average the context vectors:
Copy
Edit
([1, 0, 0, 0, 0] + [0, 0, 1, 0, 0]) / 2 = [0.5, 0, 0.5, 0, 0]
Step 4: Pass through embedding layer → get vector
(this is the learned weight matrix, W)

Step 5: Output is softmax over vocab:
Model tries to predict: "queen"


--------------------------------------------------------------------------------------------

2) >>>> Skipgram  ---word2vec >>>> 

when should we apply CBOW or skipgram ...

small dataset >>>>> CBOW 

huge dataset >>>> skipgram


# if you want to increase or improve SBOW or skipgram , how can you basically do it ? 

1) increasing the training dataset 

2) increase the windows size which  in leads to increase of vector dimension

______________________________________________________________________________________________

Google word2vec >>>>

3 billion words >>>

feature represintation of 300 dimensions vectors 


cricket >>> is alwayes there in the news 


--------------------------------------------------------------------------------------------------

# Advantages of Word2vec:

a) Sparce Matrix >>>>>> Dence Matrix 
*A dense matrix is a matrix where most values are non-zero.

b) Sementic Info is getting captured ...
[HOnset , good]   

c) past > vocabelary size >>> now >> fixed set of dimesions 

google word2vec [300 dimensions] 



d) oov >> oov is also solved 


---------------------------------------------------------------------------====================