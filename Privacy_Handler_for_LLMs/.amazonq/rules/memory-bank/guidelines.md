# Privacy Handler for LLMs - Development Guidelines

## Code Quality Standards

### Code Formatting and Structure
- **Consistent Indentation**: Use 4 spaces for Python indentation, maintain consistent spacing throughout
- **Line Length**: Keep lines under 100 characters where possible, break long lines logically
- **Import Organization**: Group imports by standard library, third-party, and local modules with blank lines between groups
- **Function/Class Spacing**: Use 2 blank lines before class definitions, 1 blank line before method definitions
- **Docstring Standards**: Use triple-quoted docstrings with clear descriptions, parameters, and return values

### Naming Conventions
- **Classes**: PascalCase (e.g., `PIIPrivacyHandler`, `ComprehensivePIIModel`, `ImageAnalyzerEngine`)
- **Functions/Methods**: snake_case (e.g., `detect_pii_entities`, `analyze_dependency`, `process_query`)
- **Variables**: snake_case (e.g., `masked_text`, `entity_type`, `confidence_score`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `COMPUTATION_KEYWORDS`, `PII_PATTERNS`, `DEFAULT_THRESHOLD`)
- **Private Methods**: Prefix with underscore (e.g., `_build_patterns`, `_generate_fake_value`, `_parse_ocr_kwargs`)

### Documentation Standards
- **Comprehensive Docstrings**: Every public method includes purpose, parameters, return values, and examples
- **Inline Comments**: Use emoji prefixes for visual organization (🔍, 🧠, 🎭, 📊, ⚖️, 💾)
- **Type Hints**: Extensive use of typing annotations (`Dict[str, Any]`, `List[Dict]`, `Tuple[bool, str]`, `Optional[str]`)
- **Error Context**: Include descriptive error messages with context about what failed and why

## Architectural Patterns

### Modular Component Design
- **Single Responsibility**: Each class handles one primary concern (detection, analysis, anonymization, API integration)
- **Dependency Injection**: Components accept dependencies through constructors for testability
- **Interface Segregation**: Clear separation between detection engines, analyzers, and handlers
- **Factory Pattern**: Used for creating different types of recognizers and processors

### Error Handling and Resilience
- **Graceful Degradation**: System continues operating with fallback implementations when components fail
- **Exception Wrapping**: Catch specific exceptions and provide meaningful error messages
- **Logging Integration**: Comprehensive logging with different levels (INFO, WARNING, ERROR)
- **Fallback Mechanisms**: Multiple detection methods with automatic failover

### Configuration Management
- **Environment Variables**: Sensitive configuration through `.env` files and environment variables
- **Default Values**: Sensible defaults for all configuration options
- **Validation**: Input validation with clear error messages for invalid configurations
- **Flexible Initialization**: Support for different initialization patterns and optional dependencies

## Implementation Patterns

### Data Processing Patterns
- **Pipeline Architecture**: Sequential processing through detection → analysis → masking → response
- **Batch Processing**: Support for processing multiple entities or queries efficiently
- **Caching Strategies**: Intelligent caching of models, patterns, and generated fake data for consistency
- **Lazy Loading**: Load expensive resources (models, APIs) only when needed

### API Integration Patterns
- **Retry Logic**: Automatic retry with exponential backoff for external API calls
- **Circuit Breaker**: Fail-fast pattern for unreliable external services
- **Timeout Handling**: Appropriate timeouts for all external service calls
- **Response Validation**: Validate API responses before processing

### Testing and Validation Patterns
- **Parametrized Testing**: Extensive use of pytest parametrization for comprehensive test coverage
- **Fixture-Based Testing**: Reusable test fixtures for common test data and configurations
- **Mock Integration**: Proper mocking of external dependencies for isolated unit testing
- **Assertion Helpers**: Custom assertion functions for domain-specific validations

## Security and Privacy Patterns

### Data Protection
- **Minimal Data Exposure**: Only preserve PII when computationally necessary
- **Secure Defaults**: Default to masking/anonymization unless explicitly required
- **Consistent Replacement**: Use caching to ensure consistent fake data generation within sessions
- **Input Sanitization**: Validate and sanitize all user inputs before processing

### Privacy-First Design
- **Context-Aware Processing**: Analyze query intent to determine PII necessity
- **Granular Control**: Fine-grained control over what gets masked vs. preserved
- **Audit Trails**: Comprehensive logging of privacy decisions and reasoning
- **Compliance Support**: Built-in support for GDPR, CCPA, and other privacy regulations

## Performance Optimization Patterns

### Efficient Processing
- **Early Termination**: Stop processing when sufficient confidence is reached
- **Batch Operations**: Process multiple entities together when possible
- **Memory Management**: Efficient memory usage with proper cleanup of large objects
- **Algorithmic Efficiency**: Use appropriate data structures and algorithms for performance

### Caching and Memoization
- **Pattern Caching**: Cache compiled regex patterns for reuse
- **Model Caching**: Keep trained models in memory to avoid repeated loading
- **Result Caching**: Cache expensive computations with appropriate invalidation strategies
- **Connection Pooling**: Reuse connections for external API calls

## Code Organization Patterns

### File Structure
- **Logical Grouping**: Related functionality grouped in appropriate modules
- **Clear Separation**: Distinct separation between core logic, utilities, and integrations
- **Consistent Naming**: File names reflect their primary purpose and contents
- **Import Hierarchy**: Clear dependency hierarchy with minimal circular dependencies

### Class Design
- **Composition Over Inheritance**: Prefer composition for building complex functionality
- **Interface Consistency**: Consistent method signatures across similar classes
- **State Management**: Clear separation between stateful and stateless components
- **Resource Management**: Proper initialization and cleanup of resources

## Development Workflow Patterns

### Code Quality Assurance
- **Comprehensive Testing**: Unit tests, integration tests, and end-to-end tests
- **Code Reviews**: Systematic code review process for all changes
- **Static Analysis**: Use of linting tools and static analysis for code quality
- **Documentation Updates**: Keep documentation synchronized with code changes

### Version Control
- **Meaningful Commits**: Clear, descriptive commit messages explaining the change
- **Feature Branches**: Use feature branches for development with proper merge strategies
- **Tag Releases**: Proper versioning and tagging of releases
- **Change Documentation**: Maintain changelog for tracking modifications

## Integration Patterns

### External Service Integration
- **Adapter Pattern**: Wrap external APIs with consistent internal interfaces
- **Configuration Abstraction**: Abstract configuration details from business logic
- **Health Checks**: Implement health check endpoints for monitoring service status
- **Graceful Shutdown**: Proper cleanup and shutdown procedures for all services

### Cross-Platform Compatibility
- **Platform Abstraction**: Abstract platform-specific functionality
- **Consistent APIs**: Maintain consistent API interfaces across different platforms
- **Resource Management**: Handle platform-specific resource management appropriately
- **Testing Coverage**: Ensure testing coverage across supported platforms

## Best Practices Summary

### Code Quality
- Write self-documenting code with clear variable and function names
- Use type hints extensively for better code clarity and IDE support
- Implement comprehensive error handling with meaningful error messages
- Follow consistent formatting and style guidelines throughout the codebase

### Architecture
- Design for modularity and testability from the beginning
- Implement proper separation of concerns between different components
- Use dependency injection for better testability and flexibility
- Design APIs that are easy to use correctly and hard to use incorrectly

### Security
- Always validate inputs and sanitize outputs
- Implement secure defaults and fail-safe mechanisms
- Log security-relevant events for audit purposes
- Regular security reviews and updates of dependencies

### Performance
- Profile code to identify actual bottlenecks before optimizing
- Use appropriate data structures and algorithms for the task
- Implement caching strategies where beneficial
- Monitor and measure performance in production environments