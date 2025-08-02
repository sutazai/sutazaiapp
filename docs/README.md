# SutazAI Documentation

Welcome to the comprehensive documentation for the SutazAI Task Automation System. This documentation is organized according to Rule 6 of our coding standards for clear, centralized, and structured documentation.

## 📁 Documentation Structure

### [📋 Project Overview](overview.md)
Complete project summary, purpose, goals, and architecture overview.

### [⚙️ Setup & Configuration](setup/)
Everything you need to get SutazAI running in your environment.

- **[Local Development Setup](setup/local_dev.md)** - Tools, dependencies, and development workflow
- **[Environment Configuration](setup/environments.md)** - Environment variables, secrets, and staging/production configs

### [🔧 Backend Documentation](backend/)
Comprehensive backend architecture and API documentation.

- **[System Architecture](backend/architecture.md)** - System design, flow diagrams, and component overview
- **[Authentication Flow](backend/auth_flow.md)** - Authentication, session logic, and security
- **[API Reference](backend/api_reference.md)** - Complete endpoint specifications, versioning, and response codes

### [🎨 Frontend Documentation](frontend/)
Frontend components, structure, and styling guidelines.

- **[Component Structure](frontend/component_structure.md)** - Folder/component layout and naming conventions
- **[Styling Guide](frontend/styling.md)** - CSS, design tokens, theming, and responsive design

### [🚀 CI/CD & Deployment](ci-cd/)
Automated deployment and continuous integration processes.

- **[CI/CD Pipeline](ci-cd/pipeline.md)** - Build triggers, testing, and automation workflow
- **[Deployment Process](ci-cd/deploy_process.md)** - Manual and automated deployment procedures

### [📊 System Information](system/)
Comprehensive system documentation and agent information.

- **[Agent Registry](ACCURATE_AGENT_REGISTRY.md)** - Complete list of available AI agents
- **[Practical Agents List](PRACTICAL_AGENTS_LIST.md)** - Working agents with examples
- **[System Documentation](SYSTEM_DOCUMENTATION.md)** - Technical system details

### [📈 Release Information](changelog.md)
Complete release history, patch notes, and upgrade guides.

## 🔍 Quick Navigation

### For Developers
- Start with [Local Development Setup](setup/local_dev.md)
- Review [System Architecture](backend/architecture.md)  
- Check [API Reference](backend/api_reference.md)
- Understand [Component Structure](frontend/component_structure.md)

### For DevOps Engineers
- Begin with [Environment Configuration](setup/environments.md)
- Study [CI/CD Pipeline](ci-cd/pipeline.md)
- Follow [Deployment Process](ci-cd/deploy_process.md)
- Monitor using guides in system documentation

### For Users
- Read [Project Overview](overview.md)
- Try [Quick Start Guide](../QUICK_START_GUIDE.md)
- Explore [Available Agents](PRACTICAL_AGENTS_LIST.md)
- Run example [Workflows](../workflows/)

### For System Administrators
- Review [System Architecture](backend/architecture.md)
- Configure [Authentication](backend/auth_flow.md)
- Set up [Monitoring](system/SYSTEM_DOCUMENTATION.md)
- Plan [Deployments](ci-cd/deploy_process.md)

## 📖 Documentation Standards

### Naming Conventions
- **Files**: lowercase, hyphen-separated (e.g., `local_dev.md`)
- **Directories**: lowercase, descriptive names
- **Sections**: Clear, hierarchical organization

### Content Guidelines
- **Clear Titles**: Descriptive headings and sections
- **Code Examples**: Working, tested code snippets
- **Consistent Formatting**: Standardized markdown structure
- **Regular Updates**: Documentation reflects current system state

### Contributing to Documentation
1. Follow the established structure and naming conventions
2. Include working code examples and clear explanations
3. Update related documentation when making changes
4. Test all links and code examples before submitting

## 🔗 External Resources

### Live System
- **Frontend Interface**: [http://localhost:8501](http://localhost:8501)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Status**: [http://localhost:8000/health](http://localhost:8000/health)

### Development Tools
- **Source Code**: Repository structure and development guidelines
- **Testing**: Unit, integration, and end-to-end testing guides
- **Monitoring**: System health and performance monitoring

### Community
- **Issues**: Bug reports and feature requests
- **Discussions**: Technical discussions and Q&A
- **Contributions**: How to contribute to the project

## 📋 Documentation Checklist

When adding or updating documentation:

- [ ] Content is accurate and reflects current system
- [ ] Code examples are tested and working
- [ ] Links are valid and point to correct locations
- [ ] Format follows established conventions
- [ ] Related documentation is updated
- [ ] Navigation links are included where appropriate

## 🆘 Getting Help

### Documentation Issues
- Missing information or outdated content
- Broken links or incorrect examples
- Unclear explanations or confusing structure

### Technical Support
- Review relevant documentation sections first
- Check the [changelog](changelog.md) for recent changes
- Search existing issues before creating new ones
- Provide specific details when requesting help

---

**Note**: This documentation is continuously updated to reflect the current state of the SutazAI system. For the most current information, always refer to the latest version in the main repository
