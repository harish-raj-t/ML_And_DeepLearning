# 👋 About Me  

I am currently building **AI-driven Agentic tools** that solve critical business challenges and **automate enterprise workflows**. My journey into AI began with a deep fascination for **LLMs and their inner workings**, which led me to explore and master them. Since then, **teaching machines to learn** has become my passion. I started with Traditional Machine Learning, gradually transitioning into **Deep Learning with a specialization in NLP**. I have further strengthened my expertise through coursework at **Georgia Institute of Technology**, completing:  
- **CS 7641:** Machine Learning  
- **CS 7643:** Deep Learning 

Now, I focus on **developing intelligent AI systems** leveraging **LLMs, RAG, and automation** to drive real-world impact.  

# 🚀 Professional Experience  

## I18n Keys Translation Automation System  
I developed an end-to-end AI tool automating the translation of I18N keys using GPT and Retrieval-Augmented Generation (RAG). This system:  

- **Handles translation across 20+ languages**  
- **Significantly reduces time and costs to company** compared to traditional methods  
- **Implements a comprehensive workflow**, including:  
  - **Preprocessing of localization keys**  
  - **RAG system preparation** with relevant context  
  - **Custom GPT assistants** for each language optimized for translation tasks  
  - **Rigorous validation** using both automated metrics and human verification  
- **Achieved 91% accuracy** through human validation testing  
- **Evaluates translation quality using advanced NLP metrics** (BLEU, COMET scores)  

## Test Case Generation Agent & Product Knowledge Chatbot  
I developed an AI chatbot powered by ChatGPT to automate test case generation, identify use-case dependencies, and retrieve relevant test cases using similarity search from ChromaDB. Additionally, the chatbot provides instant insights about the product by answering user queries. This system:  

- **Generates test cases dynamically** based on system changes
- **Identifies dependencies** between test cases and modules  
- **Retrieves relevant test cases** using similarity search for efficient validation  
- **Allows users to ask product-related questions** and get AI-powered responses  
- **Significantly reduces manual effort** while enhancing test coverage and product understanding  
- **Preprocessing of test case data** for structured retrieval  
- **Retrieval-Augmented Generation (RAG)** for enhanced contextual accuracy  
- **Custom GPT-based assistants** fine-tuned for test case generation and product Q&A  

# 🚀 Personal Projects  

# **1. TamilGPT** - [Link](https://github.com/harish-raj-t/ML_And_DeepLearning/blob/main/Mini%20LLMs/miniTamilLLM.ipynb)  

**TamilGPT** is a **decoder-only GPT model** trained on a Tamil language dataset. It is optimized for Tamil text generation, utilizing a **SentencePiece UUM tokenizer** and a **context length of 16 tokens**. The model is based on the **transformer architecture**, leveraging self-attention mechanisms to generate text effectively.  

## **Model Details**  

- **Model Type:** Decoder-only GPT  
- **Training Dataset:** Tamil Wikipedia language corpus  
- **Parameters:** 2,078,344 (≈2 million)  
- **Tokenizer:** SentencePiece (UUM)  
- **Vocab Size:** 5000  
- **Context Length:** 16 tokens  
- **Embedding Dimensions:** 128  
- **FFL Hidden Size:** 128 × 4  
- **Number of Layers:** 4  
- **Number of Attention Heads:** 4  
- **Attention Mechanism:** Masked Multi-Head Self-Attention  
- **Batch Size:** 256  
- **Learning Rate:** 1e-3  
- **Loss Function:** Cross-Entropy Loss  
- **Optimization Algorithm:** Adam Optimizer  
- **Epochs:** 10  
- **Device Support:** GPU (if available)  

---

# **2. Machine Translation Models**  

## **Seq2Seq Encoder-Decoder Model (Without Attention)**  
**[Notebook Link](https://github.com/harish-raj-t/ML_And_DeepLearning/blob/main/Machine%20Translation%20Tasks/MachineTranslation_With_SeqToSeq_Without_Attention.ipynb)**  

### **Architecture Explanation**  
- Uses a **Recurrent Neural Network (RNN)-based** Encoder-Decoder structure.  
- **Encoder:** Processes the input sentence token-by-token and generates a **fixed-size context vector** (hidden state) that represents the entire input.  
- **Decoder:** Takes this context vector and generates the output sentence sequentially.  
- **Limitation:** Since the encoder compresses all the input information into a **single fixed-length vector**, it struggles with long sentences.  

---

## **Seq2Seq Encoder-Decoder Model (With Bahdanau Attention)**  
**[Notebook Link](https://github.com/harish-raj-t/ML_And_DeepLearning/blob/main/Machine%20Translation%20Tasks/MachineTranslation_With_SeqToSeq_With_BahdanauAttention.ipynb)**  

### **Architecture Explanation**  
- This model enhances the previous Seq2Seq model by using **Bahdanau Attention**.  
- **Attention Mechanism:** Instead of relying on a single context vector, the decoder **attends** to different encoder hidden states at each decoding step.  
- **Key Benefits:**
  - Helps retain important details, especially in long sentences.  
  - Allows the model to **focus** on relevant parts of the input when generating each word.  

---

## **Encoder-Decoder Transformer Model**  
**[Notebook Link](https://github.com/harish-raj-t/ML_And_DeepLearning/blob/main/Machine%20Translation%20Tasks/MachineTranslation_With_Transformers_From_Scratch.ipynb)**  

### **Architecture Explanation**  
- **Fully Transformer-based model**, eliminating the need for RNNs.  
- **Encoder:** Uses self-attention to process the input sequence and produce contextualized representations.  
- **Decoder:** Uses masked self-attention and encoder-decoder attention to generate the output sequence.  
- **Key Features:**
  - Parallelizable (faster training and inference).  
  - Captures long-range dependencies better than RNNs.  
  - Used in modern translation models like **T5, BART, and mT5**.  

# 3. BERT Implemented from Scratch

BERT (Bidirectional Encoder Representations from Transformers) is a **transformer-based language model** designed for **contextual word representations**. This implementation trains **BERT-NSP (Next Sentence Prediction)** and **BERT-MLM (Masked Language Modeling)** separately, leveraging the transformer encoder architecture.

## Model Details

- **Model Type:** BERT (Bidirectional Encoder)  
- **Training Tasks:**  
  - **Masked Language Modeling (MLM)**  
  - **Next Sentence Prediction (NSP)**  
- **Tokenizer:** Custom WordPiece  
- **Vocab Size:** 10000  
- **Embedding Dimensions:** 128  
- **FFN Hidden Size:** 128*4  
- **Number of Layers:** 4
- **Number of Attention Heads:** 4  
- **Attention Mechanism:** Multi-Head Self-Attention  
- **Batch Size:** 256  
- **Learning Rate:** 1e-3  
- **Loss Function:**  
  - Cross-Entropy Loss (MLM)  
  - Binary Cross-Entropy Loss (NSP)  
- **Optimization Algorithm:** Adam Optimizer  
- **Epochs:** 10  
- **Device Support:** GPU (if available)  

## BERT Training Implementations  

### 1. BERT for Masked Language Modeling (MLM)  
- **[Implementation Link](https://github.com/harish-raj-t/ML_And_DeepLearning/blob/main/Implementing%20BERT%20From%20Scratch/BERT_Masked_Lang_Head_From_Scratch.ipynb)**  
- Trained to predict missing words in sentences based on context.  

### 2. BERT for Next Sentence Prediction (NSP)  
- **[Implementation Link](https://github.com/harish-raj-t/ML_And_DeepLearning/blob/main/Implementing%20BERT%20From%20Scratch/BERT_NSP_From_Scratch.ipynb)**  
- Trained to determine if a given sentence follows another in natural language.


# 4. Fine-Tuning BERT for Downstream Tasks  

Fine-tuning BERT involves adapting a **pre-trained BERT model** for specific tasks by training on task-specific datasets. These implementations focus on **sentiment analysis** and **low-rank adaptation (LoRA) fine-tuning**.  

## Fine-Tuning Implementations  

### 1. Fine-Tuning BERT for Tweet Disaster Sentiment Analysis  
- **[Implementation Link](https://github.com/harish-raj-t/ML_And_DeepLearning/blob/main/Finetuning%20BERT%20for%20Classification%2C%20NLI/BERT_sentiment_analysis(tweet_disaster).ipynb)**  
- **Task:** Classifies tweets as related to a disaster or not.  
- **Dataset:** Twitter Disaster Dataset  
- **Model Used:** Pre-trained BERT (bert-base-uncased model from huggingface) 
- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** Adam
- **Evaluation Metric:** F1 Score  

### 2. Fine-Tuning DistilBERT with LoRA PEFT for IMDB Sentiment Analysis  
- **[Implementation Link](https://github.com/harish-raj-t/ML_And_DeepLearning/blob/main/Finetuning%20BERT%20Using%20LoRA%20PEFT/notebook1895d28146.ipynb)**  
- **Task:** Classifies IMDB movie reviews as positive or negative.  
- **Dataset:** IMDB Sentiment Dataset  
- **Model Used:** DistilBERT (Lightweight BERT distilbert-base-uncased from huggingface)  
- **Fine-Tuning Approach:** LoRA (Low-Rank Adaptation) with PEFT (Parameter Efficient Fine-Tuning)  
- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** Adam 
- **Evaluation Metric:** Accuracy  

These implementations demonstrate **BERT fine-tuning for classification tasks**, using **standard full fine-tuning** and **LoRA for efficient fine-tuning**. 🚀  

