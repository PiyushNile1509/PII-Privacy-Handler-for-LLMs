# Privacy Handler for LLMs - Technology Stack

## Programming Languages & Frameworks

### Backend Development
- **Python 3.8+**: Primary backend language
- **Flask 3.0.0**: Web framework for REST API development
- **TensorFlow 2.18.0**: Deep learning framework for custom neural networks
- **NumPy 1.26.4**: Numerical computing and array operations
- **Pandas 2.1.0+**: Data manipulation and analysis
- **Scikit-learn 1.3.0**: Machine learning utilities and preprocessing

### Mobile Development
- **Flutter 3.8.1+**: Cross-platform mobile framework
- **Dart**: Programming language for Flutter development
- **SQLite**: Local database for offline-first capabilities

### Machine Learning & AI
- **TensorFlow/Keras**: Custom neural network architecture
- **NLTK 3.8.1**: Natural language processing toolkit
- **spaCy**: Advanced NLP and named entity recognition
- **Transformers**: Hugging Face transformer models
- **Google Generative AI 0.7.2**: Gemini API integration

## Core Dependencies & Versions

### Backend Services (`backend/requirements.txt`)
```
flask==3.0.0
flask-cors==4.0.0
requests==2.31.0
faker==19.3.0
python-dotenv==1.0.0
numpy==1.26.4
gunicorn==21.2.0
google-generativeai==0.3.2
```

### ML Model Dependencies (`Dependent_Model/requirements.txt`)
```
tensorflow>=2.18.0
numpy==1.26.4
pandas>=2.1.0
scikit-learn==1.3.0
nltk==3.8.1
regex==2023.6.3
faker==19.3.0
requests==2.31.0
google-generativeai==0.7.2
python-dotenv==1.0.0
matplotlib==3.7.2
seaborn==0.12.2
streamlit>=1.28.0
```

### Mobile App Dependencies (`PII_Privacy_Handler/pubspec.yaml`)
```yaml
flutter: sdk: flutter
flutter_markdown: ^0.7.7+1
http: ^1.5.0
intl: ^0.20.2
sqflite: ^2.3.0
path: ^1.8.3
sqflite_common_ffi: ^2.3.6
```

## Build Systems & Development Tools

### Python Environment
- **Virtual Environment**: `venv` for dependency isolation
- **Package Management**: `pip` with requirements.txt
- **Testing**: `pytest` for unit and integration testing
- **Code Quality**: `flake8`, `black` for formatting
- **Documentation**: `sphinx` for API documentation

### Flutter Development
- **Build System**: Flutter SDK build tools
- **Package Management**: `pub` package manager
- **Testing**: Flutter test framework
- **Platform Support**: Android, iOS, Web, Windows, macOS, Linux

### Containerization
- **Docker**: Multi-stage builds for production deployment
- **Docker Compose**: Local development environment orchestration
- **Base Images**: Python slim, Flutter official images

## Development Commands & Scripts

### Backend Development
```bash
# Environment setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Development server
python app.py
flask run --debug

# Production deployment
gunicorn app:app --bind 0.0.0.0:8000
```

### ML Model Training
```bash
# Model training
python train_model.py
python train_comprehensive_model.py
python train_final_unified_model.py

# Testing and evaluation
python run_tests.py
pytest tests/

# Demo and visualization
streamlit run demo.py
python demo.py test
```

### Mobile Development
```bash
# Flutter setup
flutter doctor
flutter pub get

# Development
flutter run
flutter run -d chrome  # Web development
flutter run -d windows  # Desktop development

# Building
flutter build apk
flutter build ios
flutter build web
```

### Presidio Integration
```bash
# Presidio setup
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_sm

# Docker deployment
docker-compose -f docker-compose-presidio.yml up
```

## External APIs & Services

### Google Gemini API
- **Purpose**: Large language model processing
- **Authentication**: API key-based authentication
- **Models**: gemini-pro, gemini-pro-vision
- **Rate Limits**: Configured per API key tier

### Microsoft Presidio
- **Components**: Analyzer, Anonymizer, Image Redactor
- **Languages**: Multi-language support with English primary
- **Deployment**: Docker containers or Python packages
- **Customization**: Custom recognizers and anonymizers

## Database & Storage

### Local Storage
- **SQLite**: Mobile app local database
- **File System**: Model artifacts, logs, configuration files
- **JSON**: Configuration and vocabulary files

### Cloud Storage (Optional)
- **AWS S3**: Model artifacts and training data
- **Google Cloud Storage**: Backup and distributed deployment
- **Azure Blob Storage**: Enterprise deployment option

## Security & Privacy

### Data Protection
- **Environment Variables**: Sensitive configuration management
- **API Key Management**: Secure credential storage
- **Local Processing**: Offline-first architecture for privacy
- **Encryption**: Data encryption at rest and in transit

### Compliance
- **GDPR**: European data protection compliance
- **CCPA**: California privacy law compliance
- **HIPAA**: Healthcare data protection (configurable)
- **SOC 2**: Security framework compliance

## Performance & Monitoring

### Optimization
- **Model Caching**: In-memory model loading
- **Connection Pooling**: Database and API connections
- **Asynchronous Processing**: Non-blocking operations
- **Batch Processing**: Efficient data processing

### Monitoring Tools
- **Logging**: Python logging, Flutter logging
- **Performance**: Memory profiler, psutil monitoring
- **Metrics**: Custom metrics for PII detection accuracy
- **Health Checks**: API endpoint health monitoring

## Deployment Architecture

### Development Environment
- **Local Development**: Flask dev server, Flutter hot reload
- **Testing**: Pytest, Flutter test framework
- **Debugging**: Python debugger, Flutter DevTools

### Production Environment
- **Web Server**: Gunicorn WSGI server
- **Reverse Proxy**: Nginx for load balancing
- **Containerization**: Docker multi-stage builds
- **Orchestration**: Docker Compose, Kubernetes support

### Cloud Platforms
- **AWS**: EC2, ECS, Lambda deployment options
- **Google Cloud**: Compute Engine, Cloud Run, App Engine
- **Azure**: Virtual Machines, Container Instances, App Service
- **Heroku**: Simplified deployment for smaller applications

## Version Control & CI/CD

### Source Control
- **Git**: Version control system
- **GitHub**: Repository hosting and collaboration
- **Branching**: Feature branches, main/develop workflow

### Continuous Integration
- **GitHub Actions**: Automated testing and deployment
- **Docker Hub**: Container image registry
- **Automated Testing**: Unit tests, integration tests, E2E tests