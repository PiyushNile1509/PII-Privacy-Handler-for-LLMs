# Privacy Handler for LLMs - Project Structure

## Directory Organization

### Root Level Components
```
Privacy_Handler_for_LLMs/
├── backend/                    # Flask API server and core services
├── Dependent_Model/           # Custom ML model training and inference
├── PII_Privacy_Handler/       # Flutter cross-platform mobile application
├── presidio-main/             # Microsoft Presidio integration and extensions
├── docs/                      # Research papers and documentation
└── deployment/                # Docker and cloud deployment configurations
```

## Core Components & Relationships

### 1. Backend Services (`/backend/`)
**Purpose**: RESTful API server providing PII detection and anonymization services

**Key Files**:
- `app.py` / `app_fixed.py`: Main Flask application with API endpoints
- `pii_dependency_handler.py`: Core PII detection and context analysis logic
- `gemini_handler.py`: Google Gemini LLM integration
- `faker_masking.py`: Synthetic data generation for anonymization
- `model_wrapper.py`: ML model inference wrapper

**Architecture**: Microservices-based with modular components for detection, analysis, and anonymization

### 2. Machine Learning Core (`/Dependent_Model/`)
**Purpose**: Custom neural network training, inference, and PII intelligence

**Structure**:
```
Dependent_Model/
├── src/                       # Core ML modules
│   ├── privacy_handler.py     # Main orchestrator
│   ├── pii_detector.py        # Custom PII detection
│   ├── model.py               # Neural network architecture
│   ├── data_processor.py      # Dataset processing
│   └── gemini_client.py       # LLM API integration
├── models/                    # Trained models and artifacts
├── tests/                     # Unit and integration tests
└── training scripts/          # Model training pipelines
```

**Relationships**: Provides trained models and inference capabilities to backend services

### 3. Mobile Application (`/PII_Privacy_Handler/`)
**Purpose**: Cross-platform Flutter app for end-user interaction

**Structure**:
```
PII_Privacy_Handler/
├── lib/
│   ├── services/              # API communication and data services
│   ├── models/                # Data models and DTOs
│   ├── widgets/               # Reusable UI components
│   ├── constants/             # App configuration and constants
│   └── screens/               # UI screens (chat, history, settings)
├── android/                   # Android-specific configurations
├── ios/                       # iOS-specific configurations
└── web/                       # Web deployment assets
```

**Relationships**: Consumes backend API services, provides offline-first capabilities

### 4. Presidio Integration (`/presidio-main/`)
**Purpose**: Microsoft Presidio framework integration for enterprise-grade PII detection

**Key Components**:
- `presidio-analyzer/`: Text analysis and PII detection engine
- `presidio-anonymizer/`: Data anonymization and transformation
- `presidio-image-redactor/`: Image-based PII detection and redaction
- `e2e-tests/`: End-to-end testing suite

**Relationships**: Integrated as a detection engine within the hybrid PII detection system

## Architectural Patterns

### 1. Layered Architecture
```
Presentation Layer    → Flutter Mobile App, Web Interface
API Layer            → Flask REST API, Authentication
Business Logic       → PII Detection, Context Analysis, Anonymization
Data Layer           → ML Models, Configuration, Logs
Integration Layer    → Presidio, Gemini API, External Services
```

### 2. Microservices Design
- **Detection Service**: PII entity identification
- **Analysis Service**: Context and dependency analysis
- **Anonymization Service**: Data masking and synthetic generation
- **LLM Service**: Large language model integration
- **Storage Service**: Conversation history and model artifacts

### 3. Hybrid Processing Pipeline
```
Input Text → Multi-Modal Detection → Context Analysis → 
Selective Masking → LLM Processing → Response Analysis → 
Data Restoration → Final Output
```

## Data Flow Architecture

### 1. Request Processing Flow
1. **Input Reception**: Mobile app or API receives user query
2. **PII Detection**: Hybrid detection using regex, NER, and Presidio
3. **Context Analysis**: Neural network determines PII computational necessity
4. **Selective Masking**: Faker-based anonymization of non-essential PII
5. **LLM Processing**: Gemini API processes masked query
6. **Response Analysis**: Determines appropriate data restoration
7. **Output Generation**: Returns privacy-preserved response to user

### 2. Training Data Flow
1. **Data Collection**: Synthetic and curated PII datasets
2. **Preprocessing**: Tokenization, embedding generation, labeling
3. **Model Training**: Multi-task learning for detection and context analysis
4. **Validation**: Cross-validation and performance evaluation
5. **Deployment**: Model serialization and integration

## Integration Points

### External Services
- **Google Gemini API**: LLM processing and response generation
- **Microsoft Presidio**: Enterprise PII detection capabilities
- **Cloud Platforms**: AWS, GCP, Azure deployment support

### Internal Communication
- **REST APIs**: Backend service communication
- **SQLite**: Local data storage for mobile app
- **File System**: Model artifacts and configuration storage
- **Logging**: Centralized logging across all components

## Scalability Considerations

### Horizontal Scaling
- Containerized microservices for independent scaling
- Load balancing for API endpoints
- Database sharding for conversation history

### Performance Optimization
- Model caching and inference optimization
- Asynchronous processing for non-blocking operations
- Connection pooling for external API calls
- Offline-first mobile architecture for reduced latency