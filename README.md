
```mermaid
flowchart TD
A[Handwritten Answer Sheet Upload] --> B{OCR Processing}
B --> C[Text Extraction from Image]
C --> D[Segmentation of Questions and Answers]
D --> E{NLP-Based Answer Evaluation}
E --> F[Compare Answers Against Key or Model Responses]
F --> G{Knowledge Gap Detection}
G -->|Identify Weak Topics| H[Personalized Feedback Generation]
H --> I[Generate Learning Resources: Videos, Notes, Quizzes]
I --> J{Feedback Delivery}
J --> K[Provide Insights to Student]
K --> L{Student Engagement}
L -->|Resources Utilized| M[Improved Understanding of Topics]
L -->|Resources Not Utilized| N[Further Recommendations]
M --> O[Track Progress and Close Knowledge Gaps]
N --> P[Reinforce Learning Resources]
O --> Q[End Process]
P --> L
