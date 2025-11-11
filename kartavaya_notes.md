#AI Notes Journey start

GPT- Generative Pre-trained Transformer

Computing Industry carbon emissions exceeding those of the entire airline industry. 
https://www.ll.mit.edu/news/ai-models-are-devouring-energy-tools-reduce-consumption-are-here-if-data-centers-will-adopt

graphics processing units (GPUs),  power-hungry hardware. , power-capping to save electricity, however training model time increases

training large language models (LLMs) requires data, compute resources, and specialized talent that only a few organizations can afford. This has led to the emergence of model as a service(ai wrappers)

# What are Language Models?
“A language model encodes statistical information about one or more languages. Intuitively, this information tells us how likely a word is to appear in a given context. For example, given the context “My favorite color is __”, a language model that encodes English should predict “blue” more often than “car”.”

“The basic unit of a language model is token. A token can be a character, a word, or a part of a word (like -tion), depending on the model.2 For example, GPT-4, a model behind ChatGPT, breaks the phrase “I can’t wait to build AI applications” into nine tokens,”. “process of breaking the original text into tokens is called tokenization.”“The set of all tokens a model can work with is the model’s vocabulary. ”. “The tokenization method and vocabulary size are decided by model developers.

- Ex “How are you?”
→ ["How", "are", "you", "?"]
- Each token is given a unique ID: , this above sentence → [105, 210, 330, 12]

# why tokens and not char or words
Why LMs use tokens (not words/chars):
    Tokens = meaningful subword units (e.g., cook + ing)
    Smaller vocab → efficient training/inference.
    Handle unknown words via subword splits (e.g., chatgpt + ing).

# Types of Language models
“They differ based on what information they can use to predict a token:”

1.  masked language models
     - trained to predict missing tokens anywhere in a sequence, using the context from both before and after the missing tokens.”
     - “In essence, a masked language model is trained to be able to fill in the blank. For example, given the context, “My favorite __ is blue”, a masked language model should predict that the blank is likely “color”
     - “used for non-generative tasks such as sentiment analysis and text classification.”
     - used for code debugging where overall prev + next coding context is needed to be understood

    
2. autoregressive language models
    - “trained to predict the next token in a sequence, using only the preceding tokens. It predicts what comes next in “My favorite color is __”
    - “are the models of choice for text generation, and for this reason, they are much more popular than masked language models.”
    - image1.png under myAssets

# What is generative AI
   - “outputs of language models are open-ended. A language model can use its fixed, finite vocabulary to construct infinite possible outputs. A model that can generate open-ended outputs is called generative, hence the term generative AI.”

# How these language models are trained?
   - “language models can be trained using self-supervision, while many other models require supervision”

# What is Supervision?
   - “Supervision refers to the process of training ML algorithms using labeled data, which can be expensive and slow to obtain.”
   - “to train a fraud detection model, you use examples of transactions, each labeled with “fraud” or “not fraud”. Once the model learns from these examples, you can use this model to predict whether a transaction is fraudulent.” This is known as labelling of data.
   - “drawback of supervision is that data labeling is expensive and time-consuming. If it costs 5 cents for one person to label one image, it’d cost $50,000 to label a million images for ImageNet”
  
# Self Supervision
 - The label comes from the input data itself , we don't need to label specifically.

# What makes a language model large? Parameters
 - “A model’s size is typically measured by its number of parameters. A parameter is a variable within an ML model that is updated through the training process.7 In general, though this is not always true, the more parameters a model has, the greater its capacity to learn desired behaviors.”
 - Bigger models → need more data to perform well.
 - Training large model on small data = waste of compute.
 - Smaller model can perform as well or better on small data.

# Foundation Models
 - Historically AI research was for NLP(Natural Lang Processing) which only deals with text , Image only Models , Audio only models
 - Multimodels-> “model that can work with more than one data modality is also called a multimodal model. A generative multimodal model is also called a large multimodal model (LMM). If a language model generates the next token conditioned on text-only tokens, a multimodal model generates the next token conditioned on both text and image tokens, or whichever modalities that the model supports”
 - example image2.png 
 - Multimodal models also need lots of data.
 - Use self-/natural language supervision (auto-generate labels).
 - CLIP trained on 400M (image, text) pairs from web (no manual labels).
 - Enabled generalization across many image tasks without extra training.

#  AI engineering techniques that you can use to adapt a model
1. Prompt engineering
2.  RAG- Retrieval-augmented generation , “Using a database to supplement the instructions 
3.  finetuning - further train
resources- image3.png

# AI Agents
- AI that can plan and use tools are AI agents.

# AI Stack = 3 layers:
- Application Dev: build apps using models; focus on prompts, context, eval, UI. ( this is where we start while using AI)
- Model Dev: tools for training, finetuning, datasets, inference, eval.
- Infrastructure: manage compute, data, serving, monitoring.
- ex- image4.png


# AI Engeneering vs ML Engeneering
- AI Eng ≠ ML Eng: key differences —
- Model use: AI Eng uses pre-trained foundation models → focus on adaptation, not training.
- Compute: works with bigger models, needs efficient inference + GPU/cluster skills.
- Output: models give open-ended results → evaluation is harder.
- Overall: AI Eng = adapt + evaluate models, not build from scratch.


# Model Development , Model and Training
- “three main responsibilities: modeling and training, dataset engineering, and inference optimization”
- “tools in this category are Google’s TensorFlow, Hugging Face’s Transformers, and Meta’s PyTorch.”
- “Developing ML models requires specialized ML knowledge. It requires knowing different types of ML algorithms (such as clustering, logistic regression, decision trees, and collaborative filtering) and neural network architectures (such as feedforward, recurrent, convolutional, and transformer). It also requires understanding how a model learns, including concepts such as gradient descent, loss function, regularization, etc.”

# Pre-training a model( most cpu intensive)
- “training a model from scratch—the model weights are randomly initialized. For LLMs, pre-training often involves training a model for text completion.”
- “A small mistake during pre-training can incur a significant financial loss and set back the project significantly. Due to the resource-intensive nature of pre-training, this has become an art that only a few practice. Those with expertise in pre-training large models, however, are heavily sought after”

# Finetuning a model
- “Finetuning, on the other hand, requires updating model weights. You adapt a model by making changes to the model itself. In general, finetuning techniques are more complicated and require more data, but they can improve your model’s quality, latency, and cost significantly”
- Limitation- “adapting the model to a new task it wasn’t exposed to during training.”
- Because the model already has certain knowledge from pre-training, finetuning typically requires fewer resources (e.g., data and compute) than pre-training.”

# Dataset engineering
- “refers to curating, generating, and annotating the data needed for training and adapting AI models.
- “traditional ML engineering, most use cases are close-ended—a model’s output can only be among predefined values.”
- “For example, spam classification with only two possible outputs, “spam” and “not spam”, is close-ended.”
- “Foundation models, however, are open-ended. Annotating open-ended queries is much harder than annotating close-ended queries—it’s easier to determine whether an email is spam than to write an essay. So data annotation is a much bigger challenge for AI engineering.”
- “Many people argue that because models are now commodities, data will be the main differentiator, making dataset engineering more important than ever. 

# Inference optimization
- “means making models faster and cheaper.
- ex- image5.png

# Prompt Engeneering
- “Prompt engineering is about getting AI models to express the desirable behaviors from the input alone, without changing the model weights.”
- “By using a different prompt engineering technique, Gemini Ultra’s performance on MMLU went from 83.7% to 90.04%.”
- “Prompt engineering is not just about telling a model what to do. It’s also about giving the model the necessary context and tools to do a given task. For complex tasks with long context, you might also need to provide the model with a memory management system so that the model can keep track of its history”
- “Traditionally, ML engineering is Python-centric. Before foundation models, the most popular ML frameworks supported mostly Python APIs. Today, Python is still popular, but there is also increasing support for JavaScript APIs, with LangChain.js, Transformers.js, OpenAI’s Node library, and Vercel’s AI SDK.”

# Foundational Models

# Sampling
- “Sampling is how a model chooses an output from all possible options. It is perhaps one of the most underrated concepts in AI. ”
- “common source for training data is Common Crawl
-  Google provides a clean subset of Common Crawl called the Colossal Clean Crawled Corpus, or C4 for short.”
-  “Some teams use heuristics to filter out low-quality data from the internet. For example, OpenAI used only the Reddit links that received at least three upvotes to train GPT-2.”
-  “perform well on tasks present in the training data but not necessarily on the tasks you care about. To address this issue, it’s crucial to curate datasets that align with your specific needs”
-  a model trained with a smaller amount of high-quality data might outperform a model trained with a large amount of low-quality data.”
-  “A model’s inference latency and cost is proportional to the number of tokens in the input and response”
-  “languages like Burmese and Hindi require a lot more tokens than English or Spanish. For the MASSIVE dataset, the median token length in English is 7, but the median length in Hindi is 32, and in Burmese, it’s a whopping 72, which is ten times longer than in English.”
-  “Assuming that the time it takes to generate a token is the same in all languages, GPT-4 takes approximately ten times longer in Burmese than in English for the same content. For APIs that charge by token usage, Burmese costs ten times more than English.”
-  They can perform better in general purpose tasks but not on domain specific tasks. “This data is unlikely to be found in publicly available internet data.”“Drug discovery involves protein, DNA, and RNA data, which follow specific formats and are expensive to acquire”
-  “Domain-specific models are especially common for biomedicine, but other fields can benefit from domain-specific models too”
-  

# Modelling
- before training, devs needs to decide what model should look like? , “What architecture should it follow? How many parameters should it have? ”

# Transformer Architecture
- “based on the attention mechanism.”
- “seq2seq (sequence-to-sequence) architecture was it's precursor, this is also used in the google translation
  - “At a high level, seq2seq contains an encoder that processes inputs and a decoder that generates outputs. Both inputs and outputs are sequences of tokens, hence the name. Seq2seq uses RNNs (recurrent neural networks) as its encoder and decoder.”
  -  “In its most basic form, the encoder processes the input tokens sequentially, outputting the final hidden state that represents the input. The decoder then generates output tokens sequentially, conditioned on both the final hidden state of the input and the previously generated token. ”
  -  ex- image6.png
  -  limitations
     -  slow as need to sequentially process token wise let's say if we had 200 tokens
     -  generates output based only on the final hidden state of the input which is like  “answers about a book using the book summary. “his limits the quality of the generated outputs.”
-  “transformer architecture addresses both problems with the attention mechanism. The attention mechanism all ows the model to weigh” “the importance of different input tokens when generating each output token. This is like generating answers by referencing any page in the book.”
-  rocess all words in parallel. Use self-attention to find relationships between all words. Faster, more scalable, captures long-range dependencies easily.
-  Transformers don’t use Recurrent Neural Networks (RNNs) at all.They replaced recurrence with self-attention.
-  RNNs (Old Method):
   -  Process input sequentially (one word at a time).Use hidden states to remember previous words. Slow to train, hard with long sequences.
-  “Inference for transformer-based language models, therefore, consists of two steps:”
     1. Prefill: “processes the input tokens in parallel. This step creates the intermediate state necessary to generate the first output token. This intermediate state includes the key and value vectors for all input tokens.”
     2. Decode- “model generates one output token at a time.” , “the parallelizable nature of prefilling and the sequential aspect of decoding both motivate many optimization techniques to make language model inference cheaper and faster.
   
# What is a transformer?
- Transformers are models that understand or generate sequences (like text) by using attention — a mechanism that helps the model decide which parts of the input matter most right now.

# Embeddings
- Turning Tokens into Vectors
- Example embedding matrix (vocab × dim):
- Token	Embedding (3D example)
         How	[0.2, -0.7, 0.5] , we have taken just 3 values but larger models like gpt have 4059 values capacity. This means how corressponds to this much values, Every value captures the meaning , different context or tone of that word. 
         are	[-0.3, 0.1, 0.8]
         you	[0.9, -0.4, 0.2]
         ?	[-0.5, 0.9, 0.0]
- in real models:, Embedding dimension (d) = 512–4096 , Each token → a point in a d-dimensional semantic space

# Attention mechanism
- “attention mechanism leverages key, value, and query vectors:” , heart of transformer architecture
- leverages key values and vectors
- Every token gets turned into 3 learned vectors:
   1. Query (Q) — what this token is currently looking for (“What do I need?”)
   2. Key (K) — how this token can describe itself (“What info do I have?”)
   3. Value (V) — the actual content it carries (“Here’s my meaning.”)
- Formula example -> image7.png 
- The attention mechanism computes how much attention to give an input token by performing a dot product between the query and its key vector.
  
   # How key vector is created?
   -  embedding is a raw vector  “How” embedding → X = [0.2, -0.7, 0.5] , we can't use this directly, we want to create a Key vector (K), a new version out of this raw info specialized for attention
   - to do this we transform it via learned transformation( a matrix that the model learns during training.)
   - If your embedding has 3 numbers (dimension = 3), then it is a 3×3 matrix (because we want to transform a 3D vector into another 3D vector).
     - example image 8.png
   - we take above X and multiply by this Wk, k = x * wk , this result K is used in attention mechanism
   - This multiplication is not random, it’s the model learning how to reshape information.
  
   # 🎯 Purpose of creating K (and Q, V):
   - The model needs a way to compare tokens and decide which ones are related or relevant.
   - Raw embeddings (X) only contain static meaning (“word identity”).
   - But attention needs contextual meaning — how this word behaves in this sentence.
   - By multiplying with learned matrices (W_Q, W_K, W_V), the model projects embeddings into new “spaces” that make comparing and combining information possible.
  
   # 📊 So, the multiplication (X × W_K):
   - Changes the coordinate system of the token’s meaning.
   - Places each token in a space where attention comparisons make sense.
   - Gives each token a “learned identity” for being recognized by queries.
   - We create K, Q, and V vectors so that the model can represent and compare tokens in a meaningful way.
  
- The Key, Query, and Value transformations let the model move from static meaning (embeddings) to relational meaning — allowing it to measure “who should pay attention to whom.
   
    # Attention weight/Scores
    - same formula  image7.png
    - Once every token has its Q and K vectors:
        - Q (Query): what the current token is looking for
        - K (Key): what each previous token offers
    - The model compares every Query with every Key using a dot product → this gives a similarity score
    - Q * K = Measures how related they are , Larger value = higher similarity
    - Softmax — Turning scores into probabilities:
    - example -> image9.png
    # 🧠 Why this helps the model:
    - This process lets each word look at all other words and decide what’s important.
    - Attention is dynamic — it changes per word, per layer.
    - This is how transformers “understand context” — not by remembering order, but by comparing meanings.
    # 💬 Example in words:
    - For the sentence “How are you?”
          - When the model predicts “?”,
          - Its Q compares to K of “How”, “are”, “you”
          - Finds “you” most relevant (highest dot product)
          - Pulls info from V(“you”)
          - Generates “?” correctly.

    # Multi-head attention
    - Allow the model to look at the same sentence from multiple perspectives at once.
    - In single head we compute Q, K, V for all tokens , Do softmax , get one context vector per token, but single view might only capture one kind of relationship.
    - So we split the embedding into several smaller parts and perform attention in parallel
    # Example
    - If your embedding dimension = 4096 and you have 32 heads: 4096/32 = 128
    # what happens inside each head?
    - example image 10.png
    - example image 11.png
    - Each head learns to focus on different relationships (syntax, grammar, long-range links).
    - Combining all individual head resultls gives the model multi-dimensional contextual understanding.
























