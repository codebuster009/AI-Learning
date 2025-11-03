#AI Notes Journey start

GPT- Generative Pre-trained Transformer

Computing Industry carbon emissions exceeding those of the entire airline industry. 
https://www.ll.mit.edu/news/ai-models-are-devouring-energy-tools-reduce-consumption-are-here-if-data-centers-will-adopt

graphics processing units (GPUs),  power-hungry hardware. , power-capping to save electricity, however training model time increases

training large language models (LLMs) requires data, compute resources, and specialized talent that only a few organizations can afford. This has led to the emergence of model as a service(ai wrappers)

# What are Language Models?
“A language model encodes statistical information about one or more languages. Intuitively, this information tells us how likely a word is to appear in a given context. For example, given the context “My favorite color is __”, a language model that encodes English should predict “blue” more often than “car”.”

“The basic unit of a language model is token. A token can be a character, a word, or a part of a word (like -tion), depending on the model.2 For example, GPT-4, a model behind ChatGPT, breaks the phrase “I can’t wait to build AI applications” into nine tokens,”. “process of breaking the original text into tokens is called tokenization.”“The set of all tokens a model can work with is the model’s vocabulary. ”. “The tokenization method and vocabulary size are decided by model developers.

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























