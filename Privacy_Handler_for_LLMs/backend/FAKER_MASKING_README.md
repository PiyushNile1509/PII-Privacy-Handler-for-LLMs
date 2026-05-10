# Faker-Based PII Masking Implementation

## Overview
This backend now uses the **Faker** library to mask Personally Identifiable Information (PII) with realistic fake data instead of generic placeholders.

## Features

### Supported PII Types
- **NAME**: Full names (e.g., "John Doe" → "Sarah Johnson")
- **EMAIL**: Email addresses (e.g., "user@example.com" → "fake@domain.com")
- **PHONE**: Phone numbers (e.g., "9876543210" → "555-123-4567")
- **AADHAAR**: 12-digit Indian ID (e.g., "123456789012" → "987654321098")
- **PAN**: Indian PAN card (e.g., "ABCDE1234F" → "XYZPQ5678R")
- **SSN**: Social Security Number
- **CREDIT_CARD**: Credit card numbers
- **ADDRESS**: Street addresses
- **DATE/DOB**: Dates and date of birth
- **COMPANY**: Company names
- **CITY/COUNTRY**: Location data
- **ZIP/ZIPCODE**: Postal codes

## Implementation

### Core Components

1. **faker_masking.py**: Utility module for PII masking
   - `FakerMasking` class with pattern-based detection
   - `generate_fake()`: Generate fake data by type
   - `mask_text()`: Mask PII in text
   - `unmask_text()`: Restore original values

2. **model_wrapper.py**: Enhanced with Faker integration
   - `_generate_fake_value()`: Type-specific fake data generation
   - `_fallback_processing()`: Faker-based fallback masking
   - Integrated with existing PII detection models

3. **app.py**: Flask backend with Faker support
   - All API endpoints use Faker masking
   - Maintains replacement mappings for reconstruction

## Usage

### Basic Example
```python
from faker_masking import FakerMasking

masker = FakerMasking()

# Mask PII
text = "My email is john@example.com and phone is 9876543210"
masked, replacements, detected = masker.mask_text(text)

print(masked)  # "My email is fake@domain.com and phone is 555-123-4567"

# Unmask to restore original
original = masker.unmask_text(masked, replacements)
```

### API Integration
The masking happens automatically in the backend:

```bash
# Test endpoint
curl -X POST http://localhost:5000/api/test-pii \
  -H "Content-Type: application/json" \
  -d '{"text": "My name is John and email is john@test.com"}'
```

Response includes:
- `original`: Original text
- `masked`: Text with fake data
- `detected`: List of detected PII types
- `replacements`: Mapping of fake → original values

## Testing

Run the test script:
```bash
cd backend
python test_faker_masking.py
```

## Benefits

1. **Realistic Data**: Fake data looks authentic, maintaining context
2. **Privacy Protection**: Original PII never sent to LLM
3. **Reversible**: Can reconstruct responses with original data
4. **Comprehensive**: Supports multiple PII types
5. **Consistent**: Seeded Faker for reproducible results

## Configuration

Modify `faker_masking.py` to:
- Add new PII patterns
- Customize fake data generation
- Adjust detection rules
- Change Faker seed for different fake data

## Dependencies

```
faker==19.3.0
```

Already included in `requirements.txt`.
