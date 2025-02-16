```mermaid
flowchart TB
    A[Start] --> B[Data Collection]
    B --> C[Preprocessing]
    C --> D[Text Preprocessing (Tokenization, Lemmatization, Stopword Removal)]
    D --> E[Word Frequency Visualization]
    E --> F[Semantic Similarity Analysis]
    F --> G[Sentence-BERT Embedding]
    G --> H[Cosine Similarity Computation]
    H --> I[Similarity Score Calculation]
    I --> J[Similarity Heatmap Visualization]
    J --> K[Knowledge Gap Detection]
    K --> L[Concept Extraction]
    L --> M[Identify Missing and Extra Concepts]
    M --> N[Concept Venn Diagram Visualization]
    N --> O[Detailed Feedback Generation]
    O --> P[Feedback Generation (Similarity Score, Missing Concepts)]
    P --> Q[Word Cloud of Missing Concepts]
    Q --> R[Similarity Score Distribution Visualization]
    R --> S[Concept Network Visualization]
    S --> T[End]

    %% Data Collection Section
    B -->|Questions, Reference Answers, Student Responses| U[Collect Data]

    %% Preprocessing Section
    C -->|Preprocessed Text| V[Preprocessing Functions]
    V --> D

    %% Similarity Analysis Section
    F -->|Embedding Responses & References| W[Generate Embeddings]
    W --> H

    %% Feedback Generation Section
    O -->|Generate Feedback Based on Similarity| X[Generate Feedback for Student]

    %% Concept Network Visualization
    S -->|Concept Graph Based on Missing Concepts| Y[Plot Concept Network Graph]
```

```mermaid
graph TD
    A[Start] --> B[Data Collection]
    B --> C[Pre-Processing]
    C --> D[Text Preprocessing (Tokenization, Lemmatization, Stopwords Removal)]
    D --> E[Visualization: Word Frequencies]
    E --> F[Semantic Similarity Analysis]
    F --> G[Similarity Score Calculation using Sentence-BERT]
    G --> H[Visualization: Similarity Heatmap]
    H --> I[Knowledge Gap Detection]
    I --> J[Concept Extraction]
    J --> K[Identify Missing and Extra Concepts]
    K --> L[Visualization: Concept Venn Diagram]
    L --> M[Feedback Generation]
    M --> N[Generate Detailed Feedback (Missing Concepts, Extra Concepts, Score)]
    N --> O[Visualization: Missing Concepts Word Cloud]
    O --> P[Visualization: Similarity Score Distribution]
    P --> Q[Visualization: Concept Network Graph]
    Q --> R[End]
```