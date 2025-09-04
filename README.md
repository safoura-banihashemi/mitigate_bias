# Mitigating Bias  

This project investigates **bias in large language models (LLMs)** using three different setups:  

- **Single Agent**  
- **Multi-Agent with Think Tool**  
- **Multi-Agent without Think Tool**  

All systems are implemented with the **[CAMEL framework](https://github.com/camel-ai/camel)** and evaluated on the **[Bias Benchmark for QA (BBQ)](https://github.com/nyu-mll/BBQ)** dataset.  

We analyze model behavior using three main evaluation metrics:  

- **Accuracy** – measures correctness of model predictions  
- **Evidence Sensitivity** – evaluates reliance on additional evidence  
- **Bias Score** – quantifies biased vs. unbiased tendencies in responses  

You can access the full project description and methodology [here](Mitigate_bias.pdf).

## Dataset  

We use the **BBQ dataset** designed to expose social biases in LLMs.  

Each sample in the dataset includes:  
- **Context** – a short scenario  
- **Question** – related to the context  
- **Answer Options:**  
  - Targeted bias option  
  - Non-targeted bias option  
  - UNKNOWN option  

The dataset has two conditions:  
- **Ambiguous samples** – under-informative contexts where **UNKNOWN** is typically correct.  
- **Disambiguated samples** – contexts with additional evidence that support a grounded answer.  

## Agent System  

The agent configurations are implemented in [`Multi_agent.py`](./Multi_agent.py):  

- **Single Agent** – one reasoning agent solving tasks independently.  
- **Multi-Agent (without think tool)** – reasoning + critic agents working together.  
- **Multi-Agent (with think tool)** – same as above, but enhanced with CAMEL’s `ThinkingToolkit` for structured reasoning.  

## Collaborators  

- Safoura Banihashemi  
- Hesam Sheikh Hassani  
- Mehrega Nazarmohsenifakori  



  
