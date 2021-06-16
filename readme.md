# English to Python code generation  

The objective is to try and generate python code from an english sentence.The machine translation model implementing the [Attention is all you need paper](https://arxiv.org/abs/1706.03762) is used. 

## Input data

This data source was the base version : [raw data](https://github.com/senthilva/END/blob/main/englishtopython/english_python_data.txt)

## Data prep-processing

*  Split the file into english and python
*  process entire python content as one line (appending) - factors the tabs; new lines
*  Performed cleaning; removed comments within code ; removed outliers
*  Replaced tabs with 4 spaces
*  regex cleaned irregular spaces : 3 -> 4; 7 -> 8; 11 -> 12 

The cleaned data is : [data clean ](https://github.com/senthilva/END/blob/main/englishtopython/clean_data.txt)

## Python tokenizer

* Built a custom tokenizer using the spacy(en) as base  
* custom rules
  * **factor spaces as a token**
  * key words
  * tabs after : and in blocks
  * == , >=, <= to be treated as single token
  * handled [,],(,),{,} 

```
# Tokenizer for python

# Get all keywords
kw_dict = {}
for kw in keyword.kwlist:
    kw_dict[kw]= [{"ORTH":kw}]

# learn 4, 8 12 spaces
special_tabs = ['\\n    ','\\n        ','\\n            ']
for tab in special_tabs:
    kw_dict[tab] = [{"ORTH":tab}]
#kw_dict    

special_cases = kw_dict
infix_re = re.compile(r'''(==|>=|<=|!=|\,|\?|\:|\;|.
                          |\‘|\’|\`|\“|\”|\"|\'|~|\(|\)|\[|\])''')


def python_tokenizer(nlp):
    return Tokenizer(nlp.vocab, 
                     infix_finditer=infix_re.finditer)
                          


py_custom = python_tokenizer(spacy_en)
```
To factor spaces as a token
```
#Modified the py thokenizer to factor spaces

def tokenize_py(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    token_texts = []
    for token in py_custom(text):
       token_texts.append(token.text)
       if token.whitespace_:  # filter out empty strings
           token_texts.append(token.whitespace_)
    return token_texts
```

## Data Augmentation Strategy

The # of original samples is ~4.5 K which cause the model to overfit.There are 2 potential approaches I considered : augment english text or augment python code. As changing of english text might change the semantics/ context; considered only augmenting python code. Approach for augmenting python code
* Replace functions names and give generic names
* Replace var names and give generic names
 
```python
# Patterns to match function and variable names
func_pat = re.compile('def (?P<func_name>[\w]+?)\(')
var_pat = re.compile(r'\n\s*(?P<var_name>[\w]+?)\s*=')

# Create new dataset by regex matching function 
# and variable names and giving generic names
final_df = pd.DataFrame(columns = ['src','trg'])
for row_idx,row in out_df.iterrows():
    var_list = []
    func_list = []
    #print(row.trg)
    func_list = list(set(func_pat.findall(row.trg)))
    var_list = list(set(var_pat.findall(row.trg)))
    if var_list:
        for var_idx,var in enumerate(var_list):
            varname = "var_"+ str(var_idx)
            final_df.loc[len(final_df)] = [row.src,row.trg.replace(var,varname)]
    if func_list:
        for func_idx,func in enumerate(func_list):
            funcname = "func_"+ str(func_idx)
            final_df.loc[len(final_df)] = [row.src,row.trg.replace(func,funcname)]
```


## Loss

Built a custom loss function using weights within the crossentropy loss function. The idea was to give a higher weightage for keywords, tabs and python tokens to force it to learn the syntax.

```python
#modified the loss function
# Built a custom function
# If keyword or tab or python tokens :2
# Rest had a weight 1

py_toks = ['(',')','{','}','[',']',':',',',';',
            '+','-','*','/','|','&','<','>','=','.',
            '%','==','!=','<=','>=','~','^','**',
            '+=','-=','*=','/=','%=','/=','//']

weight_list = []
for idx,word in enumerate(TRG.vocab.itos):


  #default
  weight = 1.0 

  # keyword or tab or common tokens
  if (keyword.iskeyword(word)) or ('\n' in word) or (word in py_toks):
      weight = 2.0
  
  weight_list.append(weight)

class_weights = torch.FloatTensor(weight_list).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index = TRG_PAD_IDX)
```


## Model Training

* Used Max Length 300
* Training loss; perplexity kept improving so increased the epochs to 30

Training vs Validation Loss

![](https://github.com/senthilva/END/blob/main/englishtopython/trainloss_vs_valloss.png)

Did not change the model parameters

```
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
```

## Observations/ Learning

* The model is able to learn the python syntax pretty well : tabs; colon ; keywords
* The amount of training data is less ~4k ; so the model struggled to learn context across words.After data augmentation the # samples was ~12K.The model loss and perplexity improved
* Model is overfitting : it is able to translate a single word to entire python code Eg presence of "sigmoid" maps to entire python code
* **Potential Areas to improve**  
  * Needs to be able to understand and generate variables/function names
  * Needs to able to understand and generate **logic** - which is going to be very tough

## Random english requests to show what model has learnt

Below are some samples which shows model learnt differnce between "function" and "program; "add"  and "subtract"

```
src ="write a sigmoid function"

predicted trg = ['def', ' ', 'sigmoid', '(', 'x', ')', ':', '\n  ', 'return', ' ', '1', ' ', '/', ' ', '(1', ' ', '+', ' ', 'math.exp', '(', '-x', ')', ')', '<eos>']
def sigmoid(x):
  return 1 / (1 + math.exp(-x))


src ="write a program to add 2 numbers"

predicted trg = ['num1', ' ', '=', ' ', '1.5', '\n', 'num2', ' ', '=', ' ', '6.3', '\n', 'sum', ' ', '=', ' ', 'num1', ' ', '+', ' ', 'num2', '\n', 'print', '(', 'f', "'", 'sum', ':', ' ', '{sum}', "'", ')', '<eos>']
num1 = 1.5
num2 = 6.3
sum = num1 + num2
print(f'sum: {sum}')


src ="write a program to subtract 2 numbers"

predicted trg = ['num1', ' ', '=', ' ', '1.5', '\n', 'num2', ' ', '=', ' ', '6.3', '\n', 'sum', ' ', '=', ' ', 'num1', ' ', '-', ' ', 'num2', '\n', 'print', '(', 'f', "'", 'sub', ':', ' ', '{sum}', "'", ')', '<eos>']
num1 = 1.5
num2 = 6.3
sum = num1 - num2
print(f'sub: {sum}')


src ="write a function to subtract 2 numbers"

predicted trg = ['def', ' ', 'var_0_two_numbers', '(', 'num1', ',', ' ', 'num2', ')', ':', '\n    ', 'var_0', ' ', '=', ' ', 'num1', ' ', '-', ' ', 'num2', '\n    ', 'return', ' ', 'var_0', '<eos>']
def var_0_two_numbers(num1, num2):
    var_0 = num1 - num2
    return var_0



src ="write a program to print 5 random numbers"

predicted trg = ['import', ' ', 'random', '\n', 'var_0', ' ', '=', ' ', '[', "'", 'a', "'", ',', ' ', "'e", "'", ',', ' ', "'i", "'", ',', ' ', "'o", "'", ',', ' ', "'u", "'", ']', '\n', 'print', '(', '[', 'random.choice', '(', 'var_0', ')', ' ', 'for', ' ', 'i', ' ', 'in', ' ', 'range', '(', '5', ')', ' ', 'if', ' ', 'var_0', ' ', '%', ' ', '2', ' ', '==', ' ', '5', ')', ']', '<eos>']
import random
var_0 = ['a', 'e', 'i', 'o', 'u']
print([random.choice(var_0) for i in range(5) if var_0 % 2 == 5)]

src ="write a program to find area of a circle"

pi = 3.14
radius = float(radius)
radius = float(radius)
area = pi * radius * radius
circumference = pi * pi * radius
print(f'area of a circle {area}')
print(f'circumference of a circle {circumference}')

```
## 25 Sample Inferences from test set 


```
**************************************************Sample    : 1  **************************************************


*******Gold *******


# write a python function using list comprehension to find even numbers in a list
def find_evennumbers(input_list):
  var_0 = [var for var in input_list if var % 2 == 0]
  return var_0


*******Predicted *******
predicted trg = ['def', ' ', 'find_evennumbers', '(', 'input_list', ')', ':', '\n  ', 'list_using_comp', ' ', '=', ' ', '[var', ' ', 'for', ' ', 'var', ' ', 'in', ' ', 'input_list', ' ', 'if', ' ', 'var', ' ', '%', ' ', '2', ' ', '==', ' ', '0', ']', '\n  ', 'return', ' ', 'list_using_comp', '<eos>']


def find_evennumbers(input_list):
  list_using_comp = [var for var in input_list if var % 2 == 0]
  return list_using_comp


****************************************************************************************************


**************************************************Sample    : 2  **************************************************


*******Gold *******


# 88 write a python fuction to print the depth of a dictionary .
def dict_depth(d):
    if isinstance(d, dict):
        return 1 + (max(map(dict_depth, d.values())) if d else 0)
    return 0
dic = {'a':1, 'b': {'c': {'d': {}}}}
print(dict_depth(dic))


*******Predicted *******
predicted trg = ['def', ' ', 'dict_depth', '(', 'd', ')', ':', '\n    ', 'if', ' ', 'isinstance', '(', 'd', ',', ' ', 'dict', ')', ':', '\n        ', 'return', ' ', '1', ' ', '+', ' ', '(max', '(', 'map', '(', 'dict_depth', ',', ' ', 'd.values', '(', ')', ')', ')', '\n    ', 'return', ' ', 'if', ' ', 'd', ' ', 'else', ' ', '0', ')', '\n    ', 'return', ' ', '0', '\n', 'dic', ' ', '=', ' ', '{', "'", 'a', "'", ':', '1', ',', ' ', "'b", "'", ':', ' ', '{', "'", 'c', "'", ':', ' ', '{', "'", ':', ' ', '{}}}}', '\n', 'print', '(', 'dict_depth', '(', 'dic', ')', ')', '<eos>']


def dict_depth(d):
    if isinstance(d, dict):
        return 1 + (max(map(dict_depth, d.values()))
    return if d else 0)
    return 0
dic = {'a':1, 'b': {'c': {': {}}}}
print(dict_depth(dic))


****************************************************************************************************


**************************************************Sample    : 3  **************************************************


*******Gold *******


# wrtie a python function to solve tower of hanoi and print necessary statements
def towerofhanoi(n , source, destination, auxiliary):
    if n==1:
        print("move disk 1 from source",source,"to destination",destination)
        return
    towerofhanoi(n-1, source, auxiliary, destination)
    print("move disk",n,"from source",source,"to destination",destination)
    towerofhanoi(n-1, auxiliary, destination, source)


*******Predicted *******
predicted trg = ['def', ' ', 'func_0', '(', 'n', ')', ':', '\n    ', 'if', ' ', 'n', ' ', '==', ' ', '1', ':', '\n        ', 'return', ' ', 'n', '\n    ', 'else', ':', '\n        ', 'return', ' ', 'n*func_0', '(', 'n-1', ')', '\n\n', 'n', ' ', '=', ' ', '[2', ',', ' ', '3', ',', ' ', '4', ',', ' ', '5', ']', '\n', 'func_0', '(', '4', ',', ' ', '6', ',', ' ', '5', ']', ')', '\n\n', 'func_0', '(', 'n', ')', '\n', 'print', '(', '"', 'the', ' ', 'original', ' ', 'list', ' ', 'is', ':', ' ', '"', ',', ' ', 'str', '(', 'n', ')', ')', '\n', 'for', ' ', 'i', ' ', 'in', ' ', 'range', '(', 'n', ')', ':', '\n    ', 'for', ' ', 'j', ' ', 'in', ' ', 'range', '(', 'n', ')', ':', '\n        ', 'if', ' ', 'not', ' ', 'j', ' ', 'in', ' ', 'range', '(', 'n', ')', ':', '\n            ', 'print', '(', 'i', ')', '\n            ', 'print', '(', '"', 'move', ' ', 'disk', ' ', 'from', ' ', 'disk', ' ', '"', ',', '"', ',', ' ', 'source', ',', ' ', 'source', ',', ' ', 'destination', ')', '\n        ', 'return', ' ', 'func_0', '(', 'n-1', ')', '<eos>']


def func_0(n):
    if n == 1:
        return n
    else:
        return n*func_0(n-1)

n = [2, 3, 4, 5]
func_0(4, 6, 5])

func_0(n)
print("the original list is: ", str(n))
for i in range(n):
    for j in range(n):
        if not j in range(n):
            print(i)
            print("move disk from disk ",", source, source, destination)
        return func_0(n-1)


****************************************************************************************************


**************************************************Sample    : 4  **************************************************


*******Gold *******


# write a function to return the torque when a force f is applied at angle thea and distance for axis of rotation to place force applied is r
def func_0(force:float,theta:float,r:float)->float:
    import math
    return force*r*math.sin(theta)


*******Predicted *******
predicted trg = ['def', ' ', 'func_0', '(', 'force', ':', 'float', ',', 'theta', ':', 'float', ',', 'theta', ':', 'float', ')', '->float', ':', '\n    ', 'import', ' ', 'math', '\n    ', 'return', ' ', 'force*r*math.sin', '(', 'theta', ')', '<eos>']


def func_0(force:float,theta:float,theta:float)->float:
    import math
    return force*r*math.sin(theta)


****************************************************************************************************


**************************************************Sample    : 5  **************************************************


*******Gold *******


# 100 write a python program that checks if a string is a pallindrome
def is_palindrome(st):
    st = st.lower()
    var_1 = st[::-1]
    try:
        assert var_1 == st
        return true
    except assertionerror:
        return false


st = "nitin"
print(is_palindrome(st))


*******Predicted *******
predicted trg = ['def', ' ', 'is_palindrome', '(', 'st', ')', ':', '\n    ', 'st', ' ', '=', ' ', 'st.lower', '(', ')', '\n    ', 'st', '[', ':', ':', ':', '-1', ']', '\n    ', 'try', ':', '\n        ', 'assert', ' ', 'rev_st', ' ', '==', ' ', 'st', '\n        ', 'return', ' ', 'true', '\n    ', 'except', ' ', 'assertionerror', ':', '\n        ', 'return', ' ', 'false', '\n\n\n', 'st', ' ', '=', ' ', '"nitin', '"', '\n', 'print', '(', 'is_palindrome', '(', 'st', ')', ')', '<eos>']


def is_palindrome(st):
    st = st.lower()
    st[:::-1]
    try:
        assert rev_st == st
        return true
    except assertionerror:
        return false


st = "nitin"
print(is_palindrome(st))


****************************************************************************************************


**************************************************Sample    : 6  **************************************************


*******Gold *******


# write a python function to perform random selection from itertools.combinations_with_replacement(iterable , r )
def ravar_2dom_combivar_2atiovar_2_with_replacemevar_2t(iterable, r):
    import ravar_2dom
    pool = tuple(iterable)
    var_2 = levar_2(pool)
    ivar_2dices = sorted(ravar_2dom.choices(ravar_2ge(var_2), k=r))
    returvar_2 tuple(pool[i] for i ivar_2 ivar_2dices)


*******Predicted *******
predicted trg = ['def', ' ', 'random_combination_with_replacement', '(', 'iterable', ',', ' ', 'r', ')', ':', '\n    ', 'import', ' ', 'random', '\n    ', 'pool', ' ', '=', ' ', 'tuple', '(', 'iterable', ')', '\n    ', 'n', ' ', '=', ' ', 'len', '(', 'pool', ')', '\n    ', 'indices', ' ', '=', ' ', 'sorted', '(', 'random.choices', '(', 'range', '(', 'n', ')', ',', ' ', 'k=r', ')', ')', '\n    ', 'return', ' ', 'tuple', '(', 'pool', '[', 'i', ']', ' ', 'for', ' ', 'i', ' ', 'in', ' ', 'indices', ')', '<eos>']


def random_combination_with_replacement(iterable, r):
    import random
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.choices(range(n), k=r))
    return tuple(pool[i] for i in indices)


****************************************************************************************************


**************************************************Sample    : 7  **************************************************


*******Gold *******


# write a python function to find the union of two arrays
def func_0(a,b):
    union=list(set(a)|set(b))
    return union


*******Predicted *******
predicted trg = ['def', ' ', 'array_var_0', '(', 'a', ',', 'b', ')', ':', '\n    ', 'var_0=list', '(', 'set', '(', 'a', ')', '|set', '(', 'b', ')', ')', '\n    ', 'return', ' ', 'var_0', '<eos>']


def array_var_0(a,b):
    var_0=list(set(a)|set(b))
    return var_0


****************************************************************************************************


**************************************************Sample    : 8  **************************************************


*******Gold *******


#   given a python list , remove all occurrence of a given number from the list
num1 = 20
list1 = [5, 20, 15, 20, 25, 50, 20]

def removevalue(samplelist, val):
    return [value for value in samplelist if value != val]
var_2 = removevalue(list1, num1)
print(var_2)

# shuffle a list randomly
import random
list = [2,5,8,9,12]
random.shuffle(list)
print ("printing shuffled list ", list)


*******Predicted *******
predicted trg = ['num1', ' ', '=', ' ', '20', '\n', 'list1', ' ', '=', ' ', '[5', ',', ' ', '20', ',', ' ', '15', ',', ' ', '20', ',', ' ', '25', ',', ' ', '50', ',', ' ', '20', ']', '\n\n', 'def', ' ', 'removevalue', '(', 'samplelist', ',', ' ', 'val', ')', ':', '\n    ', 'return', ' ', '[value', ' ', 'for', ' ', 'value', ' ', 'in', ' ', 'samplelist', ' ', 'if', ' ', 'value', ' ', '!=', ' ', 'val', ']', '\n', 'reslist', ' ', '=', ' ', 'removevalue', '(', 'var_01', ',', ' ', 'num1', ')', '\n', 'print', '(', 'reslist', ')', '\n\n', '#', ' ', 'shuffle', ' ', 'a', ' ', 'var_0', ' ', 'randomly', '\n', 'import', ' ', 'random', '\n', 'list', ' ', '=', ' ', '[2', ',', '5', ',', '8', ',', '9', ',', '12', ']', '\n', 'random.shuffle', '(', 'list', ')', '\n', 'print', ' ', '(', '"', 'printing', ' ', 'shuffled', ' ', 'var_0', ' ', '"', ',', ' ', 'var_0', ')', '<eos>']


num1 = 20
list1 = [5, 20, 15, 20, 25, 50, 20]

def removevalue(samplelist, val):
    return [value for value in samplelist if value != val]
reslist = removevalue(var_01, num1)
print(reslist)

# shuffle a var_0 randomly
import random
list = [2,5,8,9,12]
random.shuffle(list)
print ("printing shuffled var_0 ", var_0)


****************************************************************************************************


**************************************************Sample    : 9  **************************************************


*******Gold *******


# write a python program to find and print the longest word in a sentence
string = "python java c c++ javascript pascal php"

print(string)

var_0 = string.split()

id_longest = 0

for i in range(1, len(var_0)):
    if len(var_0[id_longest]) < len(var_0[i]):
        id_longest = i

print(var_0[id_longest])


*******Predicted *******
predicted trg = ['sentence', ' ', '=', ' ', "'the", ' ', 'quick', ' ', 'brown', ' ', 'fox', ' ', 'jumps', ' ', 'over', ' ', 'the', ' ', 'lazy', ' ', 'dog', "'", '\n', 'words', ' ', '=', ' ', 'sentence.split', '(', "'", ' ', "'", ')', '\n', 'lengths', ' ', '=', ' ', '[str', '(', 'len', '(', 'word', ')', ')', ' ', 'for', ' ', 'word', ' ', 'in', ' ', 'words', ']', '\n', 'print', '(', "'", ' ', "'.join", '(', 'lengths', ')', ')', '<eos>']


sentence = 'the quick brown fox jumps over the lazy dog'
words = sentence.split(' ')
lengths = [str(len(word)) for word in words]
print(' '.join(lengths))


****************************************************************************************************


**************************************************Sample    : 10  **************************************************


*******Gold *******


# write a python program to print the sum of digits of a number
n = 12345
q = 0
while(n>0):
 r=n%10
 q=q+r
 n=n//10
print("sum of digits is: "+str(q))


*******Predicted *******
predicted trg = ['n', ' ', '=', ' ', 'int', '(', 'input', '(', '"', 'enter', ' ', 'number', ':', ' ', '"', ')', ')', '\n', 'digits', ' ', '=', ' ', '0', '\n', 'for', ' ', 'i', ' ', 'in', ' ', 'range', '(', '0', ',', ' ', 'n', ')', ':', '\n    ', 'if', '(', 'digits', '[', 'i', ']', ' ', '%', ' ', '2', ' ', '==', ' ', '0', ')', ':', '\n        ', 'sum', ' ', '+=', ' ', '1', '\n    ', 'return', ' ', 'var_0', '\n', 'print', '(', 'f', "'", 'number', ' ', 'of', ' ', 'digits', ':', ' ', '"', ')', '<eos>']


n = int(input("enter number: "))
digits = 0
for i in range(0, n):
    if(digits[i] % 2 == 0):
        sum += 1
    return var_0
print(f'number of digits: ")


****************************************************************************************************


**************************************************Sample    : 11  **************************************************


*******Gold *******


# write a function to get the cumulative sum of a list
def cumulative(lists):
    cu_list = []
    var_1 = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, var_1+1)]
    return cu_list[1:]


*******Predicted *******
predicted trg = ['def', ' ', 'cumulative', '(', 'lists', ')', ':', '\n    ', 'cu_list', ' ', '=', ' ', '[', ']', '\n    ', 'length', ' ', '=', ' ', 'len', '(', 'lists', ')', '\n    ', 'cu_list', ' ', '=', ' ', '[sum', '(', 'lists', '[', '0', ':', 'x', ':', '1', ']', ')', ' ', 'for', ' ', 'x', ' ', 'in', ' ', 'range', '(', '0', ',', ' ', 'length+1', ')', ']', '\n    ', 'return', ' ', 'cu_list', '[', '1', ':', ']', '\n\n\n ', '#', ' ', 'write', ' ', 'a', ' ', 'python', ' ', 'program', ' ', 'to', ' ', 'print', ' ', 'if', ' ', 'a', ' ', 'string', ' ', '"hello', '"', ' ', 'is', ' ', 'present', ' ', 'in', ' ', 'the', ' ', 'list', '\n', 'l', ' ', '=', ' ', '[1', ',', ' ', '2.0', ',', ' ', "'hello", "'", ',', "'", 'have', "'", ',', ' ', "'a", "'", ',', ' ', "'good", "'", ',', ' ', "'day", "'", ']', '\n\n', 's', ' ', '=', ' ', "'hello", "'", '\n\n', 'if', ' ', 's', ' ', 'in', ' ', 'l', ':', '\n    ', 'print', '(', 'f', "'", '{s}', ' ', 'is', ' ', 'present', ' ', 'in', ' ', 'the', ' ', 'list', "'", ')', '\n', 'else', ':', '\n    ', 'print', '(', 'f', "'", '{s}', ' ', 'is', ' ', 'not', ' ', 'present', ' ', 'in', ' ', 'the', ' ', 'list', "'", ')', '<eos>']


def cumulative(lists):
    cu_list = []
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]
    return cu_list[1:]


 # write a python program to print if a string "hello" is present in the list
l = [1, 2.0, 'hello','have', 'a', 'good', 'day']

s = 'hello'

if s in l:
    print(f'{s} is present in the list')
else:
    print(f'{s} is not present in the list')


****************************************************************************************************


**************************************************Sample    : 12  **************************************************


*******Gold *******


# write a python program to explain the generator
def f11(x):
    yield x+1
var_0=f11(8)
print(next(var_0))


*******Predicted *******
predicted trg = ['def', ' ', 'f11', '(', 'x', ')', ':', '\n    ', 'yield', ' ', 'x+1', '\n', 'g=f11', '(', '8', ')', '\n', 'print', '(', 'next', '(', 'g', ')', ')', '<eos>']


def f11(x):
    yield x+1
g=f11(8)
print(next(g))


****************************************************************************************************


**************************************************Sample    : 13  **************************************************


*******Gold *******


# write a python function to check and print if binary representations of two numbers are anagram
from collections import counter

def checkanagram(num1,num2):
    bin1 = bin(num1)[2:]
    var_4 = bin(num2)[2:]

    zeros = abs(len(bin1)-len(var_4))
    if (len(bin1)>len(var_4)):
         var_4 = zeros * '0' + var_4
    else:
         bin1 = zeros * '0' + bin1

    dict1 = counter(bin1)
    dict2 = counter(var_4)


    if dict1 == dict2:
         print('yes')
    else:
         print('no')


*******Predicted *******
predicted trg = ['from', ' ', 'collections', ' ', 'import', ' ', 'counter', '\n\n', 'def', ' ', 'checkanagram', '(', 'num1', ',', 'num2', ')', ':', '\n    ', 'bin1', ' ', '=', ' ', 'bin', '(', 'num1', ')', '[', '2', ':', ']', '\n    ', 'bin2', ' ', '=', ' ', 'bin', '(', 'num2', ')', '[', '2', ':', ']', '\n\n    ', 'zeros', ' ', '=', ' ', 'abs', '(', 'len', '(', 'bin1', ')', '-len', '(', 'bin2', ')', ')', '\n    ', 'if', ' ', '(len', '(', 'bin1', ')', '>len', '(', 'bin2', ')', ')', ':', '\n         ', 'bin2', ' ', '=', ' ', 'zeros', ' ', '*', ' ', "'0", "'", ' ', '+', ' ', "'0", "'", ' ', "'0", "'", ' ', '+', ' ', 'bin2', '\n    ', 'else', ':', '\n         ', 'bin1', ' ', '=', ' ', '=', ' ', 'zeros', ' ', '*', ' ', "'0", "'", ' ', '+', ' ', 'bin1', '\n\n    ', 'dict1', ' ', '=', ' ', '=', ' ', 'counter', '(', 'bin1', ')', '\n    ', 'dict2', ':', '\n         ', 'print', '(', 'bin1', ')', '<eos>']


from collections import counter

def checkanagram(num1,num2):
    bin1 = bin(num1)[2:]
    bin2 = bin(num2)[2:]

    zeros = abs(len(bin1)-len(bin2))
    if (len(bin1)>len(bin2)):
         bin2 = zeros * '0' + '0' '0' + bin2
    else:
         bin1 = = zeros * '0' + bin1

    dict1 = = counter(bin1)
    dict2:
         print(bin1)


****************************************************************************************************


**************************************************Sample    : 14  **************************************************


*******Gold *******


# 77 show all local values while one function is running
def show_locvar_1l():
    import mvar_1th
    var_1 = 10
    b = 'hello there'
    print(locvar_1ls())


*******Predicted *******
predicted trg = ['def', ' ', 'show_locvar_1l', '(', ')', ':', '\n    ', 'import', ' ', 'mvar_1th', '\n    ', 'var_1', ' ', '=', ' ', '10', '\n    ', 'b', ' ', '=', ' ', "'hello", ' ', 'there', "'", '\n    ', 'print', '(', 'locvar_1ls', '(', ')', ')', '<eos>']


def show_locvar_1l():
    import mvar_1th
    var_1 = 10
    b = 'hello there'
    print(locvar_1ls())


****************************************************************************************************


**************************************************Sample    : 15  **************************************************


*******Gold *******


# given a two list of equal size create a set such that it shows the element from both lists in the pair
firstlist = [2, 3, 4, 5, 6, 7, 8]
secondlist = [4, 9, 16, 25, 36, 49, 64]
result = zip(firstlist, secondlist)
resultset = set(result)
print(resultset)


*******Predicted *******
predicted trg = ['firstlist', ' ', '=', ' ', '[2', ',', ' ', '3', ',', ' ', '4', ',', ' ', '5', ',', ' ', '6', ',', ' ', '7', ',', ' ', '8', ']', '\n', 'secondlist', ' ', '=', ' ', '[4', ',', ' ', '9', ',', ' ', '16', ',', ' ', '25', ',', ' ', '36', ',', ' ', '49', ',', ' ', '64', ']', '\n', 'result', ' ', '=', ' ', 'zip', '(', 'firstlist', ',', ' ', 'secondlist', ')', '\n', 'var_0', ' ', '=', ' ', 'set', '(', 'result', ')', '\n', 'print', '(', 'result', ')', '<eos>']


firstlist = [2, 3, 4, 5, 6, 7, 8]
secondlist = [4, 9, 16, 25, 36, 49, 64]
result = zip(firstlist, secondlist)
var_0 = set(result)
print(result)


****************************************************************************************************


**************************************************Sample    : 16  **************************************************


*******Gold *******


# write a python function that accepts a string and calculate the number of upper case letters and lower case letters
def string_test(s):
d={"upper_case":0, "lower_case":0}
for c in s:
if c.isupper():
d["upper_case"]+=1
elif c.islower():
d["lower_case"]+=1
else:
pass
print ("original string : ", s)
print ("no. of upper case characters : ", d["upper_case"])
print ("no. of lower case characters : ", d["lower_case"])

string_test('the quick brow fox')


*******Predicted *******
predicted trg = ['def', ' ', 'string_test', '(', 's', ')', ':', '\n', 'd={', '"', 'upper_case', '"', ':', '0', ',', ' ', '"lower_case', '"', ':', '0}', '\n', 'for', ' ', 'c', ' ', 'in', ' ', 's', ':', '\n', 'if', ' ', 'c.isupper', '(', ')', ':', '\n', 'd', '[', '"', 'upper_case', '"', ']', '+=1', '\n', 'elif', ' ', 'c.islower', '(', ')', ':', '\n', 'd', '[', '"', 'lower_case', '"', ']', '+=1', '\n', 'else', ':', '\n', 'pass', '\n', 'print', ' ', '(', '"', 'original', ' ', 'string', ' ', ':', ' ', '"', ',', ' ', 'd', '[', '"', 'upper_case', '"', ']', ')', '\n', 'print', ' ', '(', '"', 'no.', ' ', 'of', ' ', 'upper', ' ', 'case', ' ', 'case', ' ', 'case', ' ', ':', ' ', '"', ',', ' ', 'd', '[', '"', ']', ')', '\n', 'print', ' ', '(', '"', 'no.', ' ', 'of', ' ', 'characters', ' ', 'case', ' ', ':', ' ', '"', ',', ' ', '"', 'no.', ' ', 'case', ' ', 'length', ' ', 'case', ' ', 'case', ' ', 'case', ' ', 'case', ' ', 'case', ' ', 'case', ' ', 'case', ' ', ':', ' ', '"', ',', ' ', 'd', '[', '"', ']', ')', '\n', 'print', ' ', '(', '"', 'lower_case', '"', 'lower_case', '"', ']', ')', '<eos>']


def string_test(s):
d={"upper_case":0, "lower_case":0}
for c in s:
if c.isupper():
d["upper_case"]+=1
elif c.islower():
d["lower_case"]+=1
else:
pass
print ("original string : ", d["upper_case"])
print ("no. of upper case case case : ", d["])
print ("no. of characters case : ", "no. case length case case case case case case case : ", d["])
print ("lower_case"lower_case"])


****************************************************************************************************


**************************************************Sample    : 17  **************************************************


*******Gold *******


# write a python program to combine two dictionary by adding values for common keys
dict1 = {'a': 12, 'for': 25, 'c': 9}
dict2 = {'geeks': 100, 'geek': 200, 'for': 300}
for key in dict2:
    if key in dict1:
        dict2[key] = dict2[key] + dict1[key]
    else:
        pass


*******Predicted *******
predicted trg = ['dict1', ' ', '=', ' ', '{', "'", 'a', "'", ':', ' ', '12', ',', ' ', "'for", "'", ':', ' ', '25', ',', ' ', "'c", "'", ':', ' ', '9}', '\n', 'dict2', ' ', '=', ' ', '{', "'", 'geeks', "'", ':', ' ', '100', ',', ' ', "'geek", "'", ':', ' ', '200', ',', ' ', "'for", "'", ':', ' ', '300}', '\n', 'for', ' ', 'key', ' ', 'in', ' ', 'dict1', ':', '\n    ', 'if', ' ', 'key', ' ', 'in', ' ', 'dict1', ':', '\n        ', 'dict2', '[', 'key', ']', ' ', '=', ' ', 'dict2', '[', 'key', ']', ' ', '+', ' ', 'dict1', '[', 'key', ']', '\n    ', 'else', ':', '\n        ', 'pass', '<eos>']


dict1 = {'a': 12, 'for': 25, 'c': 9}
dict2 = {'geeks': 100, 'geek': 200, 'for': 300}
for key in dict1:
    if key in dict1:
        dict2[key] = dict2[key] + dict1[key]
    else:
        pass


****************************************************************************************************


**************************************************Sample    : 18  **************************************************


*******Gold *******


# given a two sets find the intersection and remove those elements from the first set
firstset  = {23, 42, 65, 57, 78, 83, 29}
secondset = {57, 83, 29, 67, 73, 43, 48}

var_0 = firstset.var_0(secondset)
for item in var_0:
  firstset.remove(item)
print("first set after removing common element ", firstset)


*******Predicted *******
predicted trg = ['firstset', ' ', '=', ' ', '{23', ',', ' ', '42', ',', ' ', '65', ',', ' ', '57', ',', ' ', '78', ',', ' ', '83', ',', ' ', '29}', '\n', 'secondset', ' ', '=', ' ', '{57', ',', ' ', '83', ',', ' ', '29', ',', ' ', '67', ',', ' ', '73', ',', ' ', '43', ',', ' ', '48}', '\n\n', 'intersection', ' ', '=', ' ', 'firstset.intersection', '(', 'secondset', ')', '\n', 'for', ' ', 'item', ' ', 'in', ' ', 'intersection', ':', '\n  ', 'firstset.remove', '(', 'item', ')', '\n', 'print', '(', '"', 'first', ' ', 'set', ' ', 'after', ' ', 'removing', ' ', 'common', ' ', 'element', ' ', '"', ',', ' ', 'firstset', ')', '<eos>']


firstset = {23, 42, 65, 57, 78, 83, 29}
secondset = {57, 83, 29, 67, 73, 43, 48}

intersection = firstset.intersection(secondset)
for item in intersection:
  firstset.remove(item)
print("first set after removing common element ", firstset)


****************************************************************************************************


**************************************************Sample    : 19  **************************************************


*******Gold *******


# 30 . write a python function to find hcf or gcd and return the value
def compute_var_0(x, y):

# choose the smaller number
    if x > y:
        smaller = y
    else:
        smaller = x
    for i in range(1, smaller+1):
        if((x % i == 0) and (y % i == 0)):
            var_0 = i
    return var_0


*******Predicted *******
predicted trg = ['def', ' ', 'compute_hcf', '(', 'x', ',', ' ', 'y', ')', ':', '\n\n', '#', ' ', 'choose', ' ', 'the', ' ', 'smaller', ' ', 'number', '\n    ', 'if', ' ', 'x', ' ', '>', ' ', 'y', ':', '\n        ', 'smaller', ' ', '=', ' ', 'y', '\n    ', 'else', ':', '\n        ', 'smaller', ' ', '=', ' ', 'x', '\n    ', 'for', ' ', 'i', ' ', 'in', ' ', 'range', '(', '1', ',', ' ', 'smaller+1', ')', ':', '\n        ', 'if', '(', '(', 'x', ' ', '%', ' ', 'i', ' ', '==', ' ', '0', ')', ' ', 'and', ' ', '(y', ' ', '%', ' ', 'i', ' ', '==', ' ', '0', ')', ')', ':', '\n            ', 'hcf', ' ', '=', ' ', 'i', '\n    ', 'return', ' ', 'hcf', '\n\n', 'num1', ' ', '=', ' ', '54', '\n', 'num2', ' ', '=', ' ', '24', '\n\n', 'print', '(', '"', 'the', ' ', 'h.c.f.', ' ', 'is', '"', ',', ' ', 'compute_hcf', '(', 'num1', ',', ' ', 'num2', ')', ')', '<eos>']


def compute_hcf(x, y):

# choose the smaller number
    if x > y:
        smaller = y
    else:
        smaller = x
    for i in range(1, smaller+1):
        if((x % i == 0) and (y % i == 0)):
            hcf = i
    return hcf

num1 = 54
num2 = 24

print("the h.c.f. is", compute_hcf(num1, num2))


****************************************************************************************************


**************************************************Sample    : 20  **************************************************


*******Gold *******


# write a function to check and print if a string starts with a substring using regex in python
import re


def find(string, sample) :
    if (sample in string):
        y = "^" + sample
        var_0 = re.search(y, string)
        if var_0 :
          print("string starts with the given substring")
        else :
          print("string doesn't start with the given substring")
    else :
      print("entered string isn't a substring")


*******Predicted *******
predicted trg = ['import', ' ', 're', '\n\n\n', 'def', ' ', 'find', '(', 'string', ',', ' ', 'sample', ')', ' ', ':', '\n    ', 'if', ' ', '(sample', ' ', 'in', ' ', 'string', ')', ':', '\n        ', 'y', ' ', '=', ' ', '"^', '"', ' ', '+', ' ', 'sample', '\n        ', 'x', ' ', '=', ' ', 're.search', '(', 'y', ',', ' ', 'string', ')', '\n        ', 'if', ' ', 'x', ' ', ':', '\n          ', 'print', '(', '"', 'string', ' ', 'starts', ' ', 'with', ' ', 'the', ' ', 'given', ' ', 'substring', '"', ')', '\n        ', 'else', ' ', ':', '\n          ', 'print', '(', '"', 'string', ' ', 'doesn', "'", 't', ' ', 'doesn', "'", 't', ' ', 'with', ' ', 'given', ' ', 'substring', '"', ')', '\n    ', 'else', ' ', ':', '\n      ', 'print', '(', '"', 'entered', ' ', 'string', ' ', 'string', ' ', 'isn', "'", 't', ' ', 'substring', '"', ')', '<eos>']


import re


def find(string, sample) :
    if (sample in string):
        y = "^" + sample
        x = re.search(y, string)
        if x :
          print("string starts with the given substring")
        else :
          print("string doesn't doesn't with given substring")
    else :
      print("entered string string isn't substring")


****************************************************************************************************


**************************************************Sample    : 21  **************************************************


*******Gold *******


# write   python3 code to demonstrate working of sort dictionaries list by key 's value list index using sorted ( ) + lambda ( additional parameter in case of tie )
test_list = [{"gfg" : [6, 7, 9], "is" : 9, "best" : 10},
            {"gfg" : [2, 0, 3], "is" : 11, "best" : 19},
            {"gfg" : [4, 6, 9], "is" : 16, "best" : 1}]
print("the original list : " + str(test_list))
var_2 = "gfg"
idx = 2
var_22 = "best"
res = sorted(sorted(test_list, key = lambda ele: ele[var_22]), key = lambda ele: ele[var_2][idx])
print("the required sort order : " + str(res))


*******Predicted *******
predicted trg = ['test_list', ' ', '=', ' ', '[{', '"', 'gfg', '"', ' ', ':', ' ', '[6', ',', ' ', '7', ',', ' ', '3', ']', ',', ' ', '"is', '"', ' ', ':', ' ', '9', ',', ' ', '"best', '"', ' ', ':', ' ', '10}', ',', '\n            ', '{', '"', 'gfg', '"', ' ', ':', ' ', '[4', ',', ' ', '5', ',', ' ', '6', ',', ' ', '9', ']', ',', ' ', '"is', '"', ' ', ':', ' ', '16', ',', ' ', '"best', '"', ' ', ':', ' ', '10}', ',', '\n            ', '{', '"', 'gfg', '"', ' ', ':', ' ', '[2', ',', ' ', '3', ',', ' ', '7', ',', ' ', '9', ']', ',', ' ', '"is', '"', ' ', ':', ' ', '16', ',', ' ', '11', ',', ' ', '"best', '"', ' ', ':', ' ', '19}', ',', '\n            ', '{', '"', 'gfg', '"', ' ', ':', ' ', '[4', ',', ' ', '6', ',', ' ', '9', ']', ',', ' ', '9', ']', '\n', 'k', ' ', '=', ' ', '[4', ',', ' ', '7', ',', ' ', '0', ',', ' ', '9', ']', '\n', 'res', ' ', '=', ' ', 'sorted', '(', 'test_list', ',', ' ', 'key', ' ', '=', ' ', 'lambda', ' ', 'value', ' ', 'in', ' ', 'test_list', ',', ' ', 'test_list', ')', '\n', 'print', '(', '"', 'the', ' ', 'required', ' ', 'sort', ' ', ':', ' ', '"', ' ', '+', ' ', 'str', '(', 'res', ')', ')', ')', '<eos>']


test_list = [{"gfg" : [6, 7, 3], "is" : 9, "best" : 10},
            {"gfg" : [4, 5, 6, 9], "is" : 16, "best" : 10},
            {"gfg" : [2, 3, 7, 9], "is" : 16, 11, "best" : 19},
            {"gfg" : [4, 6, 9], 9]
k = [4, 7, 0, 9]
res = sorted(test_list, key = lambda value in test_list, test_list)
print("the required sort : " + str(res)))


****************************************************************************************************


**************************************************Sample    : 22  **************************************************


*******Gold *******


# write a python program list comprehension to make a list of size n of random integers in ranges a and b
import var_3om

n = 10
a = 1
b = 100

var_3 = [var_3om.var_3int(a, b) for i in range(n)]
print(f'list : {var_3}')


*******Predicted *******
predicted trg = ['import', ' ', 'math', '\n\n', 'def', ' ', 'func_0', '(', 'a', ',', ' ', 'b', ')', ':', '\n    ', 'return', ' ', 'n', ' ', '*', ' ', '10', ' ', '+', ' ', '20', ')', '\n\n', 'def', ' ', 'func_0', '(', 'n', ')', ':', '\n    ', 'a', ' ', '=', ' ', '1', '\n    ', 'b', ' ', '=', ' ', 'n', ' ', '-', ' ', '1', ';', '\n    ', 'return', ' ', 'max', '(', 'n', ')', '\n\n', 'lst', ' ', '=', ' ', '20', ')', '\n', 'print', '(', 'func_0', '(', 'n', ')', '<eos>']


import math

def func_0(a, b):
    return n * 10 + 20)

def func_0(n):
    a = 1
    b = n - 1;
    return max(n)

lst = 20)
print(func_0(n)


****************************************************************************************************


**************************************************Sample    : 23  **************************************************


*******Gold *******


# write a python program to program to compute 1/2 + 2/3 + 3/4+ ... +n / n+1 with a given n input by console ( n>0 ) .
num = int (input ("enter number: "))
sum = 0
for i in range(num+1):
sum += float(i/(i+1))
print ("sum: {:.2f}".format(sum))


*******Predicted *******
predicted trg = ['n=int', '(', 'input', '(', '"', 'enter', ' ', 'number', ':', ' ', '"', ')', ')', '\n', 'var_0=0.0', '\n', 'for', ' ', 'i', ' ', 'in', ' ', 'range', '(', 'i+1', ',', 'n+1', ')', ':', '\n', 'print', '(', 'i', ')', '<eos>']


n=int(input("enter number: "))
var_0=0.0
for i in range(i+1,n+1):
print(i)


****************************************************************************************************


**************************************************Sample    : 24  **************************************************


*******Gold *******


# write a python function that performs selection sort on the given list or tuple or string and returns the new sorted sequence
def selection_sort(list_to_be_sorted):
    sorted_list = list_to_be_sorted[:]
    for i in range(len(sorted_list)):
        new_min = sorted_list[i]
        new_min_old_place = i
        for j in range(i+1, len(sorted_list)):
            if new_min > sorted_list[j]:
                new_min = sorted_list[j]
                new_min_old_place = j
        old_val = sorted_list[i]
        sorted_list[i] = new_min
        sorted_list[new_min_old_place] = old_val
    return sorted_list


*******Predicted *******
predicted trg = ['def', ' ', 'selection_sort', '(', 'list_to_be_sorted', ')', ':', '\n    ', 'sorted_list', ' ', '=', ' ', 'list_to_be_sorted', '[', ':', ']', '\n    ', 'for', ' ', 'i', ' ', 'in', ' ', 'range', '(', 'len', '(', 'sorted_list', ')', ')', ':', '\n        ', 'new_min', ' ', '=', ' ', 'sorted_list', '[', 'i', ']', '\n        ', 'new_min_old_place', ' ', '=', ' ', 'i', '\n        ', 'for', ' ', 'j', ' ', 'in', ' ', 'range', '(', 'i+1', ',', ' ', 'len', '(', 'sorted_list', ')', ')', ':', '\n            ', 'if', ' ', 'new_min', ' ', '>', ' ', 'sorted_list', '[', 'j', ']', ':', '\n                ', 'new_min', ' ', '=', ' ', 'sorted_list', '[', 'j', ']', '\n                ', 'new_min', ' ', '=', ' ', 'j', '\n        ', 'old_val', ' ', '=', ' ', 'sorted_list', '[', 'j', ']', '\n        ', 'sorted_list', '[', 'i', ']', ' ', '=', ' ', 'new_min', '\n        ', 'sorted_list', '[', 'new_min_old_place', ']', ' ', '=', ' ', 'old_val', '\n    ', 'return', ' ', 'sorted_list', '<eos>']


def selection_sort(list_to_be_sorted):
    sorted_list = list_to_be_sorted[:]
    for i in range(len(sorted_list)):
        new_min = sorted_list[i]
        new_min_old_place = i
        for j in range(i+1, len(sorted_list)):
            if new_min > sorted_list[j]:
                new_min = sorted_list[j]
                new_min = j
        old_val = sorted_list[j]
        sorted_list[i] = new_min
        sorted_list[new_min_old_place] = old_val
    return sorted_list


****************************************************************************************************


**************************************************Sample    : 25  **************************************************


*******Gold *******


# write a python program to print all the keys in the dictionary and store it in a list
sample_dict = {'1':1, '2':2, '3':3}
var_0 = list(sample_dict.keys())
print(f"{var_0}")


*******Predicted *******
predicted trg = ['my_list', ' ', '=', ' ', '[10', ',', ' ', '20', ',', ' ', '30', ',', ' ', '40', ',', ' ', '40', ',', ' ', '50', ',', ' ', '60', ',', ' ', '70', ']', '\n\n', 'var_0', ' ', '=', ' ', 'list', '(', 'var_0', ')', '\n', 'print', '(', 'var_0', ')', '<eos>']


my_list = [10, 20, 30, 40, 40, 50, 60, 70]

var_0 = list(var_0)
print(var_0)


****************************************************************************************************

```


## References

* [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
