# Documentation Index

Welcome to the Crypto Funding Arbitrage Platform documentation. This index will help you navigate through all available documentation.

## Quick Start

- [README](../README.md) - Project overview, features, and quick start guide
- Setup instructions
- Running the application
- Basic configuration

## Core Documentation

### 1. Architecture
**File**: [ARCHITECTURE.md](./ARCHITECTURE.md)

**Contents**:
- System overview and high-level architecture
- Backend architecture (layers, services, patterns)
- Frontend architecture (components, state management)
- Data flow and real-time communication
- Database design and entity relationships
- Security considerations
- Performance optimization
- Deployment architecture options
- Monitoring and observability
- Extensibility patterns

**Best for**: Understanding system design, adding new features, system maintenance

---

### 2. API Reference
**File**: [API_REFERENCE.md](./API_REFERENCE.md)

**Contents**:
- Complete REST API documentation
- SignalR Hub events and methods
- Request/response examples
- Error codes and handling
- Authentication (future)
- Rate limiting
- Code examples in multiple languages
- WebSocket connection management

**Best for**: Frontend developers, API consumers, integration developers

---

### 3. Deployment Guide
**File**: [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)

**Contents**:
- Development environment setup
- Docker deployment
- Cloud deployments (Azure, AWS, GCP)
- Database migration strategies
- SSL/TLS configuration
- Monitoring and logging setup
- Backup and recovery procedures
- Security checklist
- Troubleshooting production issues
- Performance tuning

**Best for**: DevOps engineers, system administrators, production deployment

---

### 4. Trading Strategy
**File**: [TRADING_STRATEGY.md](./TRADING_STRATEGY.md)

**Contents**:
- Funding rate arbitrage explained
- Strategy mechanics and setup
- Profitability analysis and calculations
- Comprehensive risk management
- Execution strategies
- Market condition analysis
- Common pitfalls and how to avoid them
- Advanced techniques
- Tax considerations
- Real-world case studies

**Best for**: Traders, strategy developers, risk managers

---

### 5. Development Guide
**File**: [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md)

**Contents**:
- Development environment setup
- Project structure walkthrough
- Backend development (adding entities, services, controllers)
- Frontend development (components, hooks, state)
- Database management and migrations
- Testing strategies (unit, integration, E2E)
- Debugging techniques
- Code style and standards
- Adding new features
- Performance optimization
- Common development issues

**Best for**: Developers contributing to the project, maintainers

---

## Documentation by Role

### For Traders

**Essential Reading**:
1. [README.md](../README.md) - Overview and setup
2. [TRADING_STRATEGY.md](./TRADING_STRATEGY.md) - Complete strategy guide
3. [API_REFERENCE.md](./API_REFERENCE.md) - Understanding the data

**Optional**:
- [ARCHITECTURE.md](./ARCHITECTURE.md) - How the system works

### For Developers

**Essential Reading**:
1. [README.md](../README.md) - Project overview
2. [ARCHITECTURE.md](./ARCHITECTURE.md) - System design
3. [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md) - Development workflow
4. [API_REFERENCE.md](./API_REFERENCE.md) - API contracts

**Optional**:
- [TRADING_STRATEGY.md](./TRADING_STRATEGY.md) - Business logic understanding
- [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) - Deployment options

### For DevOps/SysAdmins

**Essential Reading**:
1. [README.md](../README.md) - Project overview
2. [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) - Deployment procedures
3. [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture

**Optional**:
- [API_REFERENCE.md](./API_REFERENCE.md) - Health checks and monitoring
- [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md) - Troubleshooting

### For System Integrators

**Essential Reading**:
1. [API_REFERENCE.md](./API_REFERENCE.md) - Complete API documentation
2. [ARCHITECTURE.md](./ARCHITECTURE.md) - Integration points

**Optional**:
- [README.md](../README.md) - Project context
- [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) - Deployment requirements

---

## Documentation by Topic

### Getting Started
- [README.md](../README.md) - Installation and quick start
- [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) - Development environment setup

### Architecture & Design
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Complete architecture documentation
- [API_REFERENCE.md](./API_REFERENCE.md) - API design

### Development
- [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md) - Developer workflow
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Code organization

### Trading & Strategy
- [TRADING_STRATEGY.md](./TRADING_STRATEGY.md) - Complete trading guide
- [API_REFERENCE.md](./API_REFERENCE.md) - Data structures and events

### Operations & Deployment
- [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) - Complete deployment guide
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Deployment architecture

### API & Integration
- [API_REFERENCE.md](./API_REFERENCE.md) - REST API and SignalR
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Integration patterns

---

## Quick Reference

### Configuration Files

**Backend**:
- `src/CryptoArbitrage.API/appsettings.json` - Application configuration
- `src/CryptoArbitrage.API/Program.cs` - Startup configuration

**Frontend**:
- `client/vite.config.ts` - Build configuration
- `client/tailwind.config.js` - Styling configuration
- `client/.env` - Environment variables

**Docker**:
- `docker-compose.yml` - Container orchestration
- `src/CryptoArbitrage.API/Dockerfile` - Backend container
- `client/Dockerfile` - Frontend container

### Key Directories

**Backend**:
- `src/CryptoArbitrage.API/Controllers/` - API endpoints
- `src/CryptoArbitrage.API/Services/` - Business logic
- `src/CryptoArbitrage.API/Data/` - Database models
- `src/CryptoArbitrage.API/Hubs/` - SignalR hubs

**Frontend**:
- `client/src/components/` - React components
- `client/src/services/` - API clients
- `client/src/stores/` - State management
- `client/src/types/` - TypeScript types

### Important Commands

**Development**:
```bash
# Backend
dotnet run
dotnet watch run  # Hot reload

# Frontend
npm run dev

# Database
dotnet ef migrations add MigrationName
dotnet ef database update
```

**Production**:
```bash
# Docker
docker-compose up -d
docker-compose logs -f

# Build
dotnet build -c Release
npm run build
```

---

## Additional Resources

### External Links

- [.NET Documentation](https://docs.microsoft.com/en-us/dotnet/)
- [React Documentation](https://react.dev/)
- [SignalR Documentation](https://docs.microsoft.com/en-us/aspnet/core/signalr/)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Binance API](https://binance-docs.github.io/apidocs/futures/en/)
- [Bybit API](https://bybit-exchange.github.io/docs/v5/intro)

### Community

- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share ideas
- Wiki: Community-contributed guides (future)

---

## Documentation Maintenance

### Contributing to Documentation

1. **Identify gap**: Find missing or outdated information
2. **Create/update**: Write clear, concise documentation
3. **Review**: Ensure accuracy and completeness
4. **Submit PR**: Create pull request with changes

### Documentation Standards

- **Clear headings**: Use descriptive section titles
- **Code examples**: Include working code samples
- **Screenshots**: Add visuals where helpful
- **Links**: Cross-reference related documentation
- **Updates**: Keep documentation in sync with code

### Versioning

Documentation is versioned with the codebase:
- **main branch**: Latest stable version
- **develop branch**: Upcoming features
- **Tags**: Release-specific documentation

---

## Need Help?

1. **Check documentation**: Search through docs files
2. **GitHub Issues**: Search existing issues
3. **Create issue**: Report bugs or request clarification
4. **Community**: Discuss with other users

---

## Document Status

| Document | Last Updated | Status |
|----------|--------------|--------|
| README.md | 2025-01-15 | ✅ Complete |
| ARCHITECTURE.md | 2025-01-15 | ✅ Complete |
| API_REFERENCE.md | 2025-01-15 | ✅ Complete |
| DEPLOYMENT_GUIDE.md | 2025-01-15 | ✅ Complete |
| TRADING_STRATEGY.md | 2025-01-15 | ✅ Complete |
| DEVELOPMENT_GUIDE.md | 2025-01-15 | ✅ Complete |

---

## License

Documentation is licensed under the same terms as the main project (MIT License).

---

**Last Updated**: January 15, 2025
**Version**: 1.0.0
**Maintainer**: Development Team
