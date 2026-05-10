# Faker Masking - Quick Usage Guide

## Overview
Faker library is now integrated for realistic PII masking throughout the project.

## Files Created/Modified

### New Files:
1. **faker_masking.py** - Core utility for Faker-based masking
2. **test_faker_masking.py** - Unit tests for masking utility
3. **test_faker_integration.py** - Integration tests with model wrapper

### Modified Files:
1. **model_wrapper.py** - Integrated FakerMasking class for fallback processing

## Quick Start

### 1. Direct Usage (faker_masking.py)
```python
from faker_masking import FakerMasking

masker = FakerMasking()
text = "My email is john@example.com and phone is 9876543210"
masked, replacements, detected = masker.mask_text(text)

print(masked)  # "My email is fake@domain.com and phone is 5551234567"
print(detected)  # ['EMAIL', 'PHONE']

# Restore original
original = masker.unmask_text(masked, replacements)
```

### 2. Via Model Wrapper (Automatic)
```python
from model_wrapper import get_model_wrapper

wrapper = get_model_wrapper()
result = wrapper.process_query("My name is Alice, email alice@test.com")

print(result['masked_query'])  # PII masked with fake data
print(result['detected_entities'])  # List of detected PII types
print(result['final_response'])  # Response with original PII restored
```

### 3. Via Flask API (Automatic)
```bash
curl -X POST http://localhost:5000/api/test-pii \
  -H "Content-Type: application/json" \
  -d '{"text": "My email is john@test.com"}'
```

## Supported PII Types

| Type | Example Original | Example Fake |
|------|-----------------|--------------|
| EMAIL | john@example.com | fake@domain.org |
| PHONE | 9876543210 | 5551234567 |
| AADHAAR | 123456789012 | 987654321098 |
| PAN | ABCDE1234F | XYZPQ5678R |
| SSN | 123-45-6789 | 555-12-3456 |
| CREDIT_CARD | 1234 5678 9012 3456 | 4532 1234 5678 9010 |
| ADDRESS | 123 Main Street | 456 Oak Avenue |
| NAME | John Doe | Sarah Johnson |
| DATE/DOB | 1990-01-01 | 1985-05-15 |
| COMPANY | Acme Corp | Tech Solutions Inc |
| CITY | New York | Los Angeles |
| COUNTRY | USA | Canada |
| ZIP | 12345 | 90210 |

## Testing

### Run Unit Tests
```bash
cd backend
python test_faker_masking.py
```

### Run Integration Tests
```bash
cd backend
python test_faker_integration.py
```

## Configuration

### Change Faker Seed (for different fake data)
```python
masker = FakerMasking(seed=123)  # Different seed = different fake data
```

### Add Custom PII Pattern
Edit `faker_masking.py`:
```python
self.patterns = {
    'CUSTOM_ID': r'\b[A-Z]{3}\d{6}\b',  # Add your pattern
    # ... existing patterns
}
```

Then add generator:
```python
generators = {
    'CUSTOM_ID': lambda: self.fake.bothify('???######'),
    # ... existing generators
}
```

## Benefits

1. **Realistic Data**: Fake data looks authentic (e.g., real email format)
2. **Privacy Protection**: Original PII never exposed to LLM
3. **Reversible**: Can reconstruct responses with original data
4. **Comprehensive**: Supports 13+ PII types
5. **Consistent**: Seeded for reproducible results
6. **Easy Integration**: Works seamlessly with existing code

## Architecture

```
User Input → FakerMasking.mask_text() → Masked Text → LLM
                    ↓
              Replacements Map
                    ↓
LLM Response → FakerMasking.unmask_text() → Final Response
```

## Dependencies

Already included in requirements.txt:
```
faker==19.3.0
```

## Notes

- Faker seed is set to 42 by default for consistency
- All masking happens before LLM processing
- Original PII is stored in replacements map for reconstruction
- Fallback processing uses Faker when model is unavailable
