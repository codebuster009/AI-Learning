# AI Notes Journey start

# GPT- Generative Pre-trained Transformer

Computing Industry carbon emissions exceeding those of the entire airline industry.
https://www.ll.mit.edu/news/ai-models-are-devouring-energy-tools-reduce-consumption-are-here-if-data-centers-will-adopt

graphics processing units (GPUs), power-hungry hardware. , power-capping to save electricity, however training model time increases

training large language models (LLMs) requires data, compute resources, and specialized talent that only a few organizations can afford. This has led to the emergence of model as a service(ai wrappers)

# What are Language Models?

“A language model encodes statistical information about one or more languages. Intuitively, this information tells us how likely a word is to appear in a given context. For example, given the context “My favorite color is __”, a language model that encodes English should predict “blue” more often than “car”.”

“The basic unit of a language model is token. A token can be a character, a word, or a part of a word (like -tion), depending on the model.2 For example, GPT-4, a model behind ChatGPT, breaks the phrase “I can’t wait to build AI applications” into nine tokens,”. “process of breaking the original text into tokens is called tokenization.”“The set of all tokens a model can work with is the model’s vocabulary. ”. “The tokenization method and vocabulary size are decided by model developers.

Ex “How are you?”
→ ["How", "are", "you", "?"]

Each token is given a unique ID: , this above sentence → [105, 210, 330, 12]

# why tokens and not char or words

Why LMs use tokens (not words/chars):
Tokens = meaningful subword units (e.g., cook + ing)
Smaller vocab → efficient training/inference.
Handle unknown words via subword splits (e.g., chatgpt + ing).

# How these language models are trained?

“language models can be trained using self-supervision, while many other models require supervision”

# What is Supervision?

“Supervision refers to the process of training ML algorithms using labeled data, which can be expensive and slow to obtain.”

“to train a fraud detection model, you use examples of transactions, each labeled with “fraud” or “not fraud”. Once the model learns from these examples, you can use this model to predict whether a transaction is fraudulent.” This is known as labelling of data.

“drawback of supervision is that data labeling is expensive and time-consuming. If it costs 5 cents for one person to label one image, it’d cost $50,000 to label a million images for ImageNet”

# Self Supervision

The label comes from the input data itself , we don't need to label specifically.

# What is generative AI

“outputs of language models are open-ended. A language model can use its fixed, finite vocabulary to construct infinite possible outputs. A model that can generate open-ended outputs is called generative, hence the term generative AI.”

# What makes a language model large? Parameters

“A model’s size is typically measured by its number of parameters. A parameter is a variable within an ML model that is updated through the training process.7 In general, though this is not always true, the more parameters a model has, the greater its capacity to learn desired behaviors.”

Bigger models → need more data to perform well.

Training large model on small data = waste of compute.

Smaller model can perform as well or better on small data.
Ex- if someone says 7b params = 7b tiny numbers 
If each number uses 2 bytes, then:
1 parameter = 2 bytes
7 billion parameters = 7 billion × 2 bytes
Which equals:
👉 14 billion bytes And 14 billion bytes ≈ 14 gigabytes (GB) of memory.
If a model requires 14GB, your GPU must have:
✔ at least 14GB of VRAM
# RAM vs VRAM (GPU memory)
| Memory Type             | Used By | Purpose                                           |
| ----------------------- | ------- | ------------------------------------------------- |
| **RAM** (system memory) | CPU     | Runs programs, stores data for general use        |
| **VRAM** (on the GPU)   | GPU     | Runs AI models, graphics, and heavy parallel math |

Note: There are workarounds if you don't have GPU then the AI model can use sytem RAM and use system CPU but it's very slow 
# Offloading
Some tools let you split the model:
Part in VRAM
Part in RAM, Still slower, but better than CPU-only.

# Why we need GPU to run AI models?
A GPU (Graphics Processing Unit) is a special part of your computer originally designed to draw graphics (like video games). But today GPUs are used for AI because they are extremely good at doing many tiny calculations at the same time.
CPU = a few super-smart workers
GPU = thousands of simple workers working in parallel
AI models need millions of tiny math operations at once, so GPUs are perfect for that.

# Foundation Models

Historically AI research was for NLP(Natural Lang Processing) which only deals with text , Image only Models , Audio only models

Multimodels-> “model that can work with more than one data modality is also called a multimodal model. A generative multimodal model is also called a large multimodal model (LMM). If a language model generates the next token conditioned on text-only tokens, a multimodal model generates the next token conditioned on both text and image tokens, or whichever modalities that the model supports”

example image2.**png**

Multimodal models also need lots of data.

Use self-/natural language supervision (auto-generate labels).

CLIP trained on 400M (image, text) pairs from web (no manual labels).

Enabled generalization across many image tasks without extra training.

# Sampling

“Sampling is how a model chooses an output from all possible options. It is perhaps one of the most underrated concepts in AI. ”

“common source for training data is Common Crawl

Google provides a clean subset of Common Crawl called the Colossal Clean Crawled Corpus, or C4 for short.”

“Some teams use heuristics to filter out low-quality data from the internet. For example, OpenAI used only the Reddit links that received at least three upvotes to train GPT-2.”

“perform well on tasks present in the training data but not necessarily on the tasks you care about. To address this issue, it’s crucial to curate datasets that align with your specific needs”

a model trained with a smaller amount of high-quality data might outperform a model trained with a large amount of low-quality data.”

“A model’s inference latency and cost is proportional to the number of tokens in the input and response”

“languages like Burmese and Hindi require a lot more tokens than English or Spanish. For the MASSIVE dataset, the median token length in English is 7, but the median length in Hindi is 32, and in Burmese, it’s a whopping 72, which is ten times longer than in English.”

“Assuming that the time it takes to generate a token is the same in all languages, GPT-4 takes approximately ten times longer in Burmese than in English for the same content. For APIs that charge by token usage, Burmese costs ten times more than English.”

They can perform better in general purpose tasks but not on domain specific tasks. “This data is unlikely to be found in publicly available internet data.”“Drug discovery involves protein, DNA, and RNA data, which follow specific formats and are expensive to acquire”

“Domain-specific models are especially common for biomedicine, but other fields can benefit from domain-specific models too”

# AI Agents

AI that can plan and use tools are AI agents.

AI Stack = 3 layers:

Application Dev: build apps using models; focus on prompts, context, eval, UI. ( this is where we start while using AI)

Model Dev: tools for training, finetuning, datasets, inference, eval.

Infrastructure: manage compute, data, serving, monitoring.

ex- image4.png

# AI Engeneering vs ML Engeneering

AI Eng ≠ ML Eng: key differences —

Model use: AI Eng uses pre-trained foundation models → focus on adaptation, not training.

Compute: works with bigger models, needs efficient inference + GPU/cluster skills.

Output: models give open-ended results → evaluation is harder.

Overall: AI Eng = adapt + evaluate models, not build from scratch.

AI engineering techniques that you can use to adapt a model

Prompt engineering

RAG- Retrieval-augmented generation , “Using a database to supplement the instructions

# finetuning - further train
resources- image3.png

# Prompt Engeneering

“Prompt engineering is about getting AI models to express the desirable behaviors from the input alone, without changing the model weights.”

“By using a different prompt engineering technique, Gemini Ultra’s performance on MMLU went from 83.7% to 90.04%.”

“Prompt engineering is not just about telling a model what to do. It’s also about giving the model the necessary context and tools to do a given task. For complex tasks with long context, you might also need to provide the model with a memory management system so that the model can keep track of its history”

“Traditionally, ML engineering is Python-centric. Before foundation models, the most popular ML frameworks supported mostly Python APIs. Today, Python is still popular, but there is also increasing support for JavaScript APIs, with LangChain.js, Transformers.js, OpenAI’s Node library, and Vercel’s AI SDK.”

# Model Development , Model and Training

“three main responsibilities: modeling and training, dataset engineering, and inference optimization”

“tools in this category are Google’s TensorFlow, Hugging Face’s Transformers, and Meta’s PyTorch.”

“Developing ML models requires specialized ML knowledge. It requires knowing different types of ML algorithms (such as clustering, logistic regression, decision trees, and collaborative filtering) and neural network architectures (such as feedforward, recurrent, convolutional, and transformer). It also requires understanding how a model learns, including concepts such as gradient descent, loss function, regularization, etc.”

# Dataset engineering

“refers to curating, generating, and annotating the data needed for training and adapting AI models.

“traditional ML engineering, most use cases are close-ended—a model’s output can only be among predefined values.”

“For example, spam classification with only two possible outputs, “spam” and “not spam”, is close-ended.”

“Foundation models, however, are open-ended. Annotating open-ended queries is much harder than annotating close-ended queries—it’s easier to determine whether an email is spam than to write an essay. So data annotation is a much bigger challenge for AI engineering.”

“Many people argue that because models are now commodities, data will be the main differentiator, making dataset engineering more important than ever.

# Inference optimization

“means making models faster and cheaper.

ex- image5.png

# Modelling

before training, devs needs to decide what model should look like? , “What architecture should it follow? How many parameters should it have? ”

# Matrices

A matrix is a table of numbers — like many vectors stacked on top of each other.

[ 0.2 -1.1 0.5 ] ← cat

[ -0.3 0.8 1.6 ] ← dog

[ 1.2 0.4 -0.7 ] ← car

Storing all embeddings as a single matrix is memory-efficient and fast for batch operations.

# Vectors

List of numbers describing somethingheight = 170 weight = 60 age = 22 --> [170, 60, 22]

Because computers understand numbers, not words. So vectors let computers: find similar words compare meanings do math with meaning store information efficiently

# Embedding Vector

It is a big table of vectors where each row represents a token (word, subword, character, etc.).

The model stores an embedding matrix E of shape (V, d_model):

V = vocab size (number of different tokens). If the model knows 50,000 tokens, then V = 50,000.

d_model = embedding dimension (e.g. 512, 768, 1024, 12288). , If each embedding is a vector of 512 real numbers, then d_model = 512.

so, embedding matrix is: E → a matrix with V rows and d_model columns, This means: Each row = 1 token, Each row is a vector of size d_model

Example Image13.png

# Embeddings

Turning Tokens into Vectors

Example embedding matrix (vocab × dim):

Token Embedding (3D example)
How [0.2, -0.7, 0.5] , we have taken just 3 values but larger models like gpt have 4059 values capacity. This means how corressponds to this much values, Every value captures the meaning , different context or tone of that word.
are [-0.3, 0.1, 0.8]
you [0.9, -0.4, 0.2]
? [-0.5, 0.9, 0.0]

in real models:, Embedding dimension (d) = 512–4096 , Each token → a point in a d-dimensional semantic space

Index Lookup for an token

embedding for token id i is simply row i of E: e_i = E[i]. , this is an index lookup, not a dot product

# Note

A Transformer needs both token meaning and token position.

Token meaning comes from the token embedding matrix E.

Position information comes from the positional embedding matrix Pos (or P).

For each token at position p with token ID i, the model looks up:

E[i] → the token embedding vector

Pos[p] → the positional embedding vector

we get, X_p = E[i] + Pos[p], giving the model both meaning and order. This is just the process of creating one vector for 1 token.

# WHAT IS and Why DO WE NEED POSITIONAL ENCODING?

Transformers don’t know word order.

Ex- dog bites man and man bites dog

So we must inject position information , This is called positional encoding.
TWO WAYS:

1. Sinusoidal Positional Encoding (fixed)

uses math (sine and cosine waves).

For each position p (0, 1, 2, 3…) And each dimension i (0…d_model−1) It computes a number.

No learning. It’s deterministic.

2. Learned Positional Embeddings (trainable)

just another matrix, like token embeddings.

Number of rows = max_pos , Number of columns = d_model

If max_pos = 512, the model can handle sentences up to 512 words/tokens long.

So the matrix has 512 rows, one row for position 0, one for position 1, … up to 511.

Example Image14.png

Stacking all X_p in one structure to get final X

X = [ X_0 X_1 X_2]

X is the set of all token vectors (one per word), stacked into a table. The transformer uses this table as the input to attention.

# What does the Transformer do with X_p?

takes X_p and passes it through three learned linear layers.

W_Q → Query weight matrix

W_K → Key weight matrix

W_V → Value weight matrix

Creating Q(query), K(key), V(value)

X_p (the embedding vector) is multiplied into Q, K, V

# How these learned layers are created? Ans: Training

The cat sat on the ___ , ANS: MAT , but model gives banana in starting, After realization it adjusts all its internal numbers (including W_Q, W_K, W_V) to reduce this mistake.

How these(How are W_Q, W_K, W_V updated?) are corrected? ANS: Backpropagation

# Backpropagation 
is a method where the model asks: “How much did each weight contribute to the error?”, Then each weight gets nudged in the direction that will reduce future errors.

The model calculates how wrong it was.

“Hey W_Q, if you had made Q vectors different, maybe the model would pay attention to the right words.”

“Hey W_K, your keys didn’t help the model focus on the right places.”

Hey W_V, your values didn’t carry the right information.”

# Gradient descent?

taking tiny steps to reduce the loss.

If output is wrong, gradient descent says:

“W_Q should change slightly this way…”

“W_K should change slightly that way…”

“W_V should change slightly…”

These are just numbers the model learns during training.

new_weight = old_weight - learning_rate × gradient (Learning rate = how big your step is. Gradient = direction to change.)

# What is neural network?
A big collection of tiny math units connected together, working together to solve a task.

A neural network uses artificial neurons — very simple math functions that:

take numbers in

do some math

output a number

They’re “neural” because they act a bit like brain neurons.

# Transformer Architecture

“based on the attention mechanism.”

“seq2seq (sequence-to-sequence) architecture was it's precursor, this is also used in the google translation

“At a high level, seq2seq contains an encoder that processes inputs and a decoder that generates outputs. Both inputs and outputs are sequences of tokens, hence the name. Seq2seq uses RNNs (recurrent neural networks) as its encoder and decoder.”

Easy way-> Recurrent neural network sees somthing adds in notebook one by one
Ex- Bank work might have diff meanings depending on context ex- River Bank , Loan from Bank, So whole context needs to be known

“In its most basic form, the encoder processes the input tokens sequentially, outputting the final hidden state that represents the input. The decoder then generates output tokens sequentially, conditioned on both the final hidden state of the input and the previously generated token. ”

ex- image6.png

# limitations

slow as need to sequentially process token wise let's say if we had 200 tokens

generates output based only on the final hidden state of the input which is like “answers about a book using the book summary. “his limits the quality of the generated outputs.”

so it's like asking detail questions but you can only answer a whole chapter using summary. Some important details will get lost.

“transformer architecture addresses both problems with the attention mechanism. The attention mechanism all ows the model to weigh” “the importance of different input tokens when generating each output token. This is like generating answers by referencing any page in the book.”

process all words/tokens in parallel. Use self-attention to find relationships between all words. Faster, more scalable, captures long-range dependencies easily.

Transformers don’t use Recurrent Neural Networks (RNNs) at all.They replaced recurrence with self-attention.

# Self Attention
The model is paying attention to the words in the same sentence (“self”),

Not to some outside information.

For each word, the model:

Looks at all the other words

Decides how relevant they are

Gives more weight (“attention”) to the important ones

Combines the information to better understand the word

# RNNs (Old Method): Recurrent Neural Network

Process input sequentiaslly (one word at a time).Use hidden states to remember previous words. Slow to train, hard with long sequences.

# “Inference(conclusion) for transformer-based language models, therefore, consists of two steps:”

1. Prefill: “processes the input tokens in parallel. This step creates the intermediate state necessary to generate the first output token. This intermediate state includes the key and value vectors for all input tokens.”

2. Decode- “model generates one output token at a time.” , “the parallelizable nature of prefilling and the sequential aspect of decoding both motivate many optimization techniques to make language model inference cheaper and faster.

# What is a transformer?

Transformers are models that understand or generate sequences (like text) by using attention — a mechanism that helps the model decide which parts of the input matter most right now.

# What are dimensions in a model?
we say larger dimensions, we are talking about things like:
hidden size (embedding dimension)
number of heads
size of each head
feedforward dimension
number of layers
These define how big the model is inside.
If you increase: embedding dimension → model gets more parameters number of layers → model gets more parameters feedforward dimension → more parameters
heads → more parameters
These are like adding: more neurons, more connections, more weights
👉 More dimensions = more weights = more parameters = larger model

# WHAT IS CONTEXT LENGTH?
Context length = how many tokens the model can read at once.
Newer models have 8,192… 32,000… 100,000… even 1,000,000
But WHY doesn’t this increase parameters?
Because context length does not add any new weights to the model.
Instead, it changes how much memory the model uses during a single inference, not the model's actual size.

# Why context length does NOT add parameters
Because the same attention block weights are reused for each position in the sequence.
Self-attention does this:
For each token, compute Q, K, V using the SAME learned matrices
So whether you have: 100 tokens 1000 tokens 10,000 tokens 1,000,000 tokens
The number of matrices (parameters) stays the same.
The model just has to do more computation for more tokens.

# Attention mechanism

“attention mechanism leverages key, value, and query vectors:” , heart of transformer architecture

leverages key values and vectors

Every token gets turned into 3 learned vectors:

Query (Q) — what this token is currently looking for (“What do I need?”)

Key (K) — how this token can describe itself (“What info do I have?”)

Value (V) — the actual content it carries (“Here’s my meaning.”)

Formula example -> image7.png

The attention mechanism computes how much attention to give an input token by performing a dot product between the query and its key vector.

# How key vector is created?

embedding is a raw vector “How” embedding → X = [0.2, -0.7, 0.5] , we can't use this directly, we want to create a Key vector (K), a new version out of this raw info specialized for attention

to do this we transform it via learned transformation( a matrix that the model learns during training.)

If your embedding has 3 numbers (dimension = 3), then it is a 3×3 matrix (because we want to transform a 3D vector into another 3D vector).

example image 8.png

we take above X and multiply by this Wk(learned matrices), k = x * wk , this result K is used in attention mechanism

This multiplication is not random, it’s the model learning how to reshape information.

# 🎯 Purpose of creating K (and Q, V):

The model needs a way to compare tokens and decide which ones are related or relevant.

Raw embeddings (X) only contain static meaning (“word identity”).

But attention needs contextual meaning — how this word behaves in this sentence.

By multiplying with learned matrices (W_Q, W_K, W_V), the model projects embeddings into new “spaces” that make comparing and combining information possible.

📊 So, the multiplication (X × W_K):

Changes the coordinate system of the token’s meaning.

Places each token in a space where attention comparisons make sense.

Gives each token a “learned identity” for being recognized by queries.

We create K, Q, and V vectors so that the model can represent and compare tokens in a meaningful way.

The Key, Query, and Value transformations let the model move from static meaning (embeddings) to relational meaning — allowing it to measure “who should pay attention to whom.

# HOW ATTENTION SCORES ARE COMPUTED 

Once we have Q (query) and K (key) vectors, the Transformer must decide How much should token A pay attention to token B
# It does this in 3 steps:

1. Compute raw attention scores
For each pair of tokens: score = Q • K   (dot product) 
“performing a dot product between the query vector and its key vector. A high score means that the model will use more of that page’s content (its value vector) when generating the book’s summary. ”

Q and K are created by multiplying the token embedding X with learned weight matrices refer [# WHAT IS and Why DO WE NEED POSITIONAL ENCODING?]

# What is a dot product?
If
Q = [1, 2, 3]
K = [4, 5, 6]

Then
Q·K = (1×4 + 2×5 + 3×6)
   = 4 + 10 + 18 = 32

Bigger dot product → vectors point in same direction → tokens are related
Smaller dot product (0 or negative) → not related

Example Image15.png

2. Scale the score
Transformers divide the dot product by √d_model. Because if vectors are huge (like 1024 dimensions), dot products become extremely large, which makes softmax explode.
scaled_score = (Q • K) / √d_k

3. Apply Softmax
Softmax converts the scaled scores into probabilities that add up to 1.0.

Example before softmax:
[1.1, 4.7, 5.2, 2.0, 4.9]
After softmax (example numbers):
[0.02, 0.20, 0.33, 0.06, 0.29]

Now the model knows:

Pay 33% attention to “sat”
Pay 29% attention to “mat”
Pay 20% attention to “cat”
Pay almost 0% to “the”

same formula image7.png
example -> image9.png
example-> image12.png
example-image16.png

# 🧠 Why this helps the model:

This process lets each word look at all other words and decide what’s important.

Attention is dynamic — it changes per word, per layer.

This is how transformers “understand context” — not by remembering order, but by comparing meanings.

# 💬 Example in words:

For the sentence “How are you?”
- When the model predicts “?”,
- Its Q compares to K of “How”, “are”, “you”
- Finds “you” most relevant (highest dot product)
- Pulls info from V(“you”)
- Generates “?” correctly.

# Why compute is hard and why is it so hard to extend context “length for transformer models”

Every token → 1 Query, 1 Key, 1 Value vector

To compute attention for token t, we do: Qₜ compared with all previous Keys K₁..Kₜ

Gives t scores

Softmax → t attention weights

Weighted sum of V₁..Vₜ → outputₜ

So:

Token 1 → 1 comparison

Token 2 → 2 comparisons

Token L → L comparisons

Total comparisons =

1 + 2 + ... + L = O(L²)


Must store all Ks and Vs = O(L) memory

Must compute Q·K for all pairs = O(L²) compute

That’s why increasing context window is expensive.

# Multi-head attention

“The attention mechanism is almost always multi-headed. ”

Allow the model to look at the same sentence from multiple perspectives at once.

In single head we compute Q, K, V for all tokens , Do softmax , get one context vector per token, but single view might only capture one kind of relationship.

So we split the embedding into several smaller parts and perform attention in parallel

Example

For each token embedding (size = d_model), instead of creating ONE Q vector, we create many:

Q₁, Q₂, … Qₙ → one per head

K₁, K₂, … Kₙ

V₁, V₂, … Vₙ

Where n = number of heads.

For each head i:

Compare Qᵢ with all previous Kᵢ

Softmax the scores

Build weighted sum of Vᵢ

This gives one output vector per head (128 dims).

If your embedding dimension = 4096 and you have 32 heads: 4096/32 = 128


what happens inside each head?

example image 10.png

example image 11.png

Each head learns to focus on different relationships (syntax, grammar, long-range links).

Combining all individual head resultls gives the model multi-dimensional contextual understanding.

We concatenate all head outputs

After concatenation → we multiply by W_O (“output projection”).

Why?

Because:

each head learned a different view

W_O learns how to combine all views into one meaningful representation

Shape of W_O:

each head becomes a pattern detector focusing on different relationships.

example
Head 1: noun modifies adjective

Head 6: who (“he/she”) refers to

Head 10: long-range dependencies

Head 18: parentheses/brackets matching

Head 22: sentence boundaries

Head 29: style/genre

Head 31: number agreement (is/are)

Combining them produces powerful contextual understanding.

# Output Projection ( W__0)
Multi-head attention = 32 small attention modules, each doing its own job.

Concatenation = put their results side-by-side (no mixing).

Output projection = learned mixing of all heads back into 4096-D space.

In linear algebra terms:

W_O computes a learned linear combination of all head outputs.

# Linear Vs Non-Linear ( basics)
We say something to be linear if it follows two simple rules always:
1. Additivity
      - f(a + b) = f(a) + f(b)
2. Scaling
      - f(c·a) = c·f(a)
      - If you stretch a vector by a number c and then apply f, it must be the same as applying f first, then stretching the result
# Why these rules matter?
Because if BOTH rules are true, then:
👉 f cannot bend space
👉 f cannot curve anything
👉 f cannot twist one part more than another

# What are non-linear functions?
When a neural network processes information, each layer takes numbers in, does some math, and sends new numbers out.

If it only did normal math (like adding and multiplying), the network would be limited—it could only learn straight-line patterns.

Nonlinear functions are special little operations that “bend” the data so the network can learn complicated things—like images, language, or speech.

Think of them as shapes the network uses to understand the world.

It can only: stretch , shrink , rotate , flip , shear (slant) , Everything stays straight and proportional.


# A transformer can contain multiple transformer blocks, but, in general, each transformer contains:
1. Attention Module
“consists of four weight matrices: query, key, value, and output projection.”

2. MLP(multi layer perceptron) also k/a (Feedforward / Linear Layers)
part helps the model learn complex patterns after attention has combined information.

Consists of: 

Linear layer 1 → FF1
Activation function (ReLU, GELU, etc.) --> These add non-linearity (allowing the model to learn complex ideas).
Linear layer 2 → FF2

# Modules Before and After Transformer Blocks1
1. Embedding Module (before blocks)

Turns: tokens → embedding vectors , positions → positional embeddings

If a model stores positions 0–2047, its maximum context length is 2048 tokens (unless special tricks are used).

# Embedding module contains: 
Embedding matrix
Positional embedding matrix
Output Layer (after blocks)

Turns final hidden vectors into probabilities for each token in the vocabulary.

Uses one matrix (unembedding matrix).

Sometimes called the model head.

# How model size is determined?

1. Model dimension (d_model)

Determines size of Q, K, V, O matrices

Larger d_model = bigger model = more parameters

2. Number of transformer blocks

More blocks = deeper model = more parameters

3. Feedforward dimension (d_ff)

Size of hidden layer inside the MLP

Often much larger than d_model

4. Vocabulary size (V)

Larger vocab = bigger embedding & output matrices


# AI Carbon emissions
https://youtu.be/MJQIQJYxey4?si=bLWX0zODr6flRRTQ

# ReLU/Rectified Linear Unit.?
one of the simplest and most popular nonlinear functions.
Here is its entire job:
👉 If a number is negative → turn it into 0
👉 If a number is positive → keep it the same
Mathematically: ReLU(x) = max(0, x)

# Where do negative values come from?
Every neuron in a neural network does this simple math:
output = (weight * input) + bias
Bias is just a number the neuron adds after multiplying.

# Before tranining starts
Every neuron has:
a weight → random
a bias → random

Example:weight = 0.17 , bias = -0.93

During training, the model makes a prediction. The model thinks the answer should be 2.5 but the correct answer is 5. So the model has error.
The model calculates:
“How should I change the weight and the bias
so next time, my prediction is closer to the truth?” Just a small change. It does this millions of times during training. Eventually, the bias becomes a number that helps the neuron make correct predictions.

# Why does a neuron even NEED a bias?
Let’s say:

output = weight * input

This ALWAYS passes through (0,0).

But some patterns in data need lines that start higher or lower.

Examples:

If input = 0 but the correct answer is 7 → you NEED bias 7

If input = 0 but the correct answer is -3 → you NEED bias -3

Bias lets the neuron shift up or down to fit the data.

It shifts the line up or down.
Every neuron produces positives and negatives during training.
Every neuron produces an output based on an equation.That equation is a line.
y = 2x + 3 , y = -1.5x + 7
WITHOUT RELU = ONLY STRAIGHT LINES
That means:
the model cannot learn curves
it cannot adapt
it cannot grow
it cannot learn complex patterns
ex- ____/

This hinge is everything.
You can build almost any shape using enough hinges.

https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/

When you pass this “bent line” to the next layer, the next neuron works on a shape that already has structure.

# Limitations of transformer architecture

# Till above we discussed the embedding module before the transformer blocks, now let's see output layer after the transformer blocks” “called the unembedding layer. ”
“refer to the output layer as the model head, as it’s the model’s last layer before output generation.”
example: image17.png

# Some other architectures(dive deep yourself , hard to outperform transformer architechture)
⭐ Major Deep Learning Architectures (with one-line summaries)
🔹 CNN (Convolutional Neural Network) Great for images; extracts spatial patterns like edges and textures.
🔹 RNN (Recurrent Neural Network) Processes sequences one step at a time; keeps short-term memory.
🔹 LSTM (Long Short-Term Memory) Advanced RNN that handles longer dependencies using gates.
🔹 GRU (Gated Recurrent Unit) Simpler LSTM version with similar performance on many tasks.
🔹 Seq2Seq (Encoder–Decoder RNN)

# Old machine translation setup; replaced mostly by transformers.
🔹 GAN (Generative Adversarial Network) Two neural nets compete: one generates, one judges; great for images.
🔹 VAE (Variational Autoencoder) Probabilistic model for generating and compressing data.
🔹 Capsule Networks Tried to encode spatial hierarchies; never became mainstream.
⭐ Architectures directly related to Transformers / Modern LLMs
🔹 Transformer Uses self-attention for parallel sequence processing; current dominant architecture.
🔹 MoE Transformer (Mixture-of-Experts) Only activates part of the model per token → huge models with small compute.
🔹 Linear Attention Transformers Variants designed to reduce quadratic attention cost (e.g., Performer, Linformer).
🔹 Longformer / BigBird Transformers modified for long documents via sparse attention. 
🔹 Reformer Memory-efficient transformer using hashing. 
🔹 FlashAttention Transformers Optimized attention computation for speed and memory.

⭐ New / Transformer Alternatives (Rising Today)
# example Image 18.png
🔹 Mamba (State Space Model) RNN-like architecture with linear memory and long-context ability.
🔹 S4 / S5 (Structured State Space Models) Mathematically powerful SSMs for long-range sequence modeling.
🔹 Jamba (Hybrid Transformer + Mamba + MoE) Mixes attention + SSM + MoE for long sequences with efficiency.
🔹 RWKV Hybrid RNN/Transformer that trains in parallel but runs sequentially.
🔹 Samba Enhanced SSM improvement inspired by Mamba.
⭐ Graph-based and Reasoning Architectures 
🔹 GNN (Graph Neural Network) Operates on graph-structured data like molecules or social networks.
🔹 GAT (Graph Attention Network) GNN with attention weights for node importance.
⭐ Diffusion Models (for images, audio, video)
🔹 DDPM (Denoising Diffusion Probabilistic Model) Generates images by reversing a noise-adding process.
🔹 UNet (commonly used inside diffusion models) Encoder–decoder network with skip connections for image generation.
🔹 Latent Diffusion (Stable Diffusion)

# Diffusion in compressed latent space: faster and more efficient.
⭐ Memory + Retrieval Architectures
🔹 RAG (Retrieval-Augmented Generation) LLM + external knowledge search for more accurate answers.
🔹 RETRO (DeepMind) Transformer with explicit large-scale retrieval baked in.
🔹 KNN-LM
 and many more ....

 # Sparse vs Dense
sparse- [0, 0, 1.1, 0, 0, 0.4] , many 0 values
Dense model:[2.4, -1.3, 0.7, 5.2, -3.0, 1.9]
Because zeros mean:
They don’t change anything
They don’t need to be stored in full detail
They don’t need to be used when calculating the model’s output
but they can become non-zero 

# Model Size(Page 63)
Example: 7B parameters but 90% zeros
If a model has: 7 billion total parameters 90% of them are zero
Then only 10% matter:
7 billion × 0.10 = 700 million non-zero parameters
So even though it sounds like a huge model (“7B!”),
in reality it only has 700M active parameters worth computing.
Carrying a backpack
Dense model = backpack with 7 billion heavy objects
Sparse model = backpack with 7 billion objects, but 90% weigh zero
If most items weigh zero, your backpack is much lighter.
So it’s easier to carry (less compute).

# 3 numbers signal a model’s scale:
Number of parameters, which is a proxy for the model’s learning capacity.
Number of tokens a model was trained on, which is a proxy for how much a model learned.
Number of FLOPs, which is a proxy for the training cost.”

Excerpt From
AI Engineering
Chip Huyen
This material may be protected by copyright.

# Mixture of Experts a new type of sparse model
“An MoE model is divided into different groups of parameters, and each group is an expert. Only a subset of the experts is active for (used to) process each token.”
“ example, Mixtral 8x7B is a mixture of eight experts, each expert with seven billion parameters. If no two experts share any parameter, it should have 8 × 7 billion = 56 billion parameters. However, due to some parameters being shared, it has only 46.7 billion parameters.”

# Importance of size of data it was trained on
“ for model dataset sizes are measured by the number of training samples. ”
“The number of tokens isn’t a perfect measurement either, as different models can have different tokenization processes,”
“LLMs are trained using datasets in the order of trillions of tokens”
“The number of tokens in a model’s dataset isn’t the same as its number of training tokens. The number of training tokens measures the tokens that the model is trained on”

# 3 golden rules of data training
Quantity
Quality
Diversity

# Computation cost and requirements
“A more standardized unit for a model’s compute requirement is FLOP, or floating point operation. FLOP measures the number of floating point operations performed for a certain task.”
“The plural form of FLOP, FLOPs, is often confused with FLOP/s, floating point operations per Second. FLOPs measure the compute requirement for a “task, whereas FLOP/s measures a machine’s peak performance.”

Example - “example, an NVIDIA H100 NVL GPU can deliver a maximum of 60 TeraFLOP/s: 6 × 1013 FLOPs a second or 5.2 × 1018 FLOPs a day.16”
“1 FLOP/s-day = 60 × 60 × 24 = 86,400 FLOPs”
“Assume that you have 256 H100s. If you can use them at their maximum capacity and make no training mistakes, it’d take you (3.14 × 1023) / (256 × 5.2 × 1018) = ~236 days, or approximately 7.8 months, to train GPT-3-175B. , “Generally, if you can get half the advertised performance, 50% utilization, you’re doing okay.
not possible to utilize something 100 percent
“At 70% utilization and $2/h for one H100,17 training GPT-3-175B would cost over $4 million:

$2/H100/hour × 256 H100 × 24 hours × 256 days / 0.7 = $4,142,811.43















