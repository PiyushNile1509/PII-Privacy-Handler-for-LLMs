# PII Privacy Handler - Final Year Project

## Project Overview

The PII Privacy Handler is an intelligent system that dynamically masks sensitive data in user prompts while preserving essential information required for accurate responses. The system uses a custom-trained neural network to intelligently determine which PII entities to mask based on contextual necessity and restores original data in the final response for user-friendly output.

## 🚀 Key Features

- **Intelligent PII Detection**: Custom neural network trained from scratch for PII entity recognition
- **Context-Aware Masking**: Analyzes query intent to determine which PII to preserve vs. mask
- **Multi-Domain Support**: Handles PII across healthcare, finance, education, government, and more
- **Custom Word Embeddings**: Trained domain-specific embeddings for better performance
- **LLM Integration**: Seamless integration with Google's Gemini API
- **Data Restoration**: Smart restoration of original PII in responses where appropriate
- **Real-time Processing**: Fast inference for production use

## 🏗️ System Architecture

```
User Input → PII Detection → Context Analysis → Selective Masking → 
LLM Processing → Response Analysis → Data Restoration → Final Output
```

### Core Components

1. **PII Detection Module**: Custom-trained neural network for entity extraction
2. **Context Analysis Engine**: Intent understanding and necessity determination
3. **Masking & Replacement Engine**: Faker-based data substitution
4. **LLM Interface**: Gemini API integration
5. **Response Processor**: Output restoration and user-friendly formatting

## 📊 Model Architecture

- **Input Layer**: Text tokenization and embedding layer
- **Hidden Layers**: Bidirectional LSTM layers for sequence processing
- **Output Layers**: 
  - PII Entity Classification head (multi-label)
  - Context Necessity Prediction head (binary for each PII entity)
- **Custom Features**: Domain-specific word embeddings and attention mechanisms

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- 8GB+ RAM recommended
- GPU support optional but recommended

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Dependent_Model
```

2. **Create virtual environment**
```bash
python -m venv pii_privacy_env
source pii_privacy_env/bin/activate  # On Windows: pii_privacy_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download spaCy language model**
```bash
python -m spacy download en_core_web_sm
```

5. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your Gemini API key
```

## 🎯 Usage

### Training the Model

1. **Prepare the dataset**
   - Ensure `pii_100000_new.csv` is in the root directory
   - The dataset should contain columns: `query`, `masked_output`, `domain`, `pii_entities`, `replaced_pii`, `reason_for_not_replaced`

2. **Train the model**
```bash
python train_model.py
```

This will:
- Process the dataset and build vocabulary
- Train custom word embeddings
- Train the multi-task neural network
- Save the trained model, vocabulary, and embeddings
- Generate training plots and evaluation metrics

### Running the Demo

#### Streamlit Web Interface
```bash
streamlit run demo.py
```

#### Command Line Interface
```bash
python demo.py
```

#### Run Test Cases
```bash
python demo.py test
```

### Using the API

```python
from src.privacy_handler import PIIPrivacyHandler

# Initialize the handler
handler = PIIPrivacyHandler(
    model_path='models/pii_privacy_model.h5',
    vocab_path='models/vocab.json'
)

# Process a query
result = handler.process_query(
    "My name is John Smith and my phone is 555-1234. Calculate the sum of digits in my phone number."
)

print(f"Original: {result['original_query']}")
print(f"Masked: {result['masked_query']}")
print(f"Response: {result['final_response']}")
```

## 📈 Examples

### Example 1: General Conversation
**Input**: "Hi, my name is Prasad Zade and I live in Pune. How's the weather today?"
- **Analysis**: No PII needed for weather query
- **Masked Output**: "Hi, my name is John Smith and I live in Boston. How's the weather today?"
- **Reason**: Name and location not required for weather information

### Example 2: Mathematical Operations
**Input**: "My name is Prasad Zade, my phone number is 7897897456. Can you add all digits in my phone number?"
- **Analysis**: Phone number needed for calculation, name not needed
- **Masked Output**: "My name is John Smith, my phone number is 7897897456. Can you add all digits in my phone number?"
- **Reason**: Phone number preserved for mathematical operation, name masked

### Example 3: Location-Based Query
**Input**: "I'm Prasad from 123 MG Road, Pune. What are the nearest hospitals to my address?"
- **Analysis**: Address needed for location-based search, name not needed
- **Masked Output**: "I'm John from 123 MG Road, Pune. What are the nearest hospitals to my address?"
- **Reason**: Address preserved for location search, name masked

## 🔍 Supported PII Types

### Common PII
- Full Name, Date of Birth, Age, Gender, Address
- Phone Number, Email Address, Username, Password/PIN
- Nationality, Photograph, Biometric Identifiers

### Domain-Specific PII
- **Financial**: Bank Account, Credit Card, IBAN/SWIFT, Insurance Policy
- **Healthcare**: Medical Record Number, Health Insurance, Diagnosis, Prescriptions
- **Education**: Student ID, Grades/Marks, Parent/Guardian Name, School Name
- **Government**: National ID/SSN, Driver's License, Passport, Tax ID
- **Digital**: IP Address, MAC Address, Device ID, Social Media Handle

## 📊 Performance Metrics

The system is evaluated on:
- **Privacy Preservation**: Percentage of unnecessary PII successfully masked (>95% target)
- **Functional Accuracy**: Correctness when PII is preserved for computational needs (>95% target)
- **Response Quality**: User satisfaction with final output
- **Processing Speed**: End-to-end response time
- **Context Understanding**: Accuracy in determining PII necessity

## 🧪 Testing

### Unit Tests
```bash
pytest tests/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### Performance Benchmarks
```bash
python benchmark.py
```

## 📁 Project Structure

```
Dependent_Model/
├── src/
│   ├── __init__.py
│   ├── data_processor.py      # Dataset processing and vocabulary building
│   ├── model.py               # Neural network architecture
│   ├── pii_detector.py        # PII detection and context analysis
│   ├── gemini_client.py       # Gemini API integration
│   └── privacy_handler.py     # Main system orchestrator
├── models/                    # Trained models and artifacts
├── plots/                     # Training visualizations
├── logs/                      # System logs
├── tests/                     # Test suite
├── train_model.py            # Training script
├── demo.py                   # Demo application
├── requirements.txt          # Dependencies
├── .env.example             # Environment template
└── README.md                # This file
```

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini API key
- `MODEL_PATH`: Path to trained model (default: models/)
- `VOCAB_PATH`: Path to vocabulary file
- `LOG_LEVEL`: Logging level (INFO, DEBUG, ERROR)

### Model Parameters
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Training epochs (default: 50)
- `LEARNING_RATE`: Learning rate (default: 0.001)
- `EMBEDDING_DIM`: Embedding dimensions (default: 300)
- `LSTM_UNITS`: LSTM hidden units (default: 256)

## 🚀 Deployment

### Docker Deployment
```bash
docker build -t pii-privacy-handler .
docker run -p 8501:8501 pii-privacy-handler
```

### Cloud Deployment
The system can be deployed on:
- AWS EC2 with GPU support
- Google Cloud Platform
- Azure Machine Learning
- Heroku (CPU-only)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini API for LLM capabilities
- Faker library for synthetic data generation
- TensorFlow/Keras for deep learning framework
- The open-source community for various tools and libraries

## 📞 Support

For support, email [your-email@example.com] or create an issue in the repository.

## 🔮 Future Enhancements

- Support for more languages
- Advanced anonymization techniques
- Real-time streaming processing
- Integration with more LLM providers
- Enhanced privacy compliance features
- Federated learning capabilities

---

**Note**: This system is designed for educational and research purposes. Ensure compliance with relevant privacy regulations (GDPR, CCPA, etc.) when deploying in production environments.