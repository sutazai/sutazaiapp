Rule 12: Universal Deployment Script
CRITICAL: Complete End-to-End Automation - One Command Does Everything:

Zero Interruption Deployment: Script must run from start to finish without ANY manual intervention
Complete System Setup: Must install, configure, and start the entire system to fully operational state
One Command Success: ./deploy.sh --env=prod results in complete running system - NO EXCEPTIONS
No Partial Deployments: Script either completes 100% successfully or rolls back completely
No Manual Steps: Zero requirement for human intervention during any phase of deployment
Full Stack Deployment: Database, backend, frontend, networking, monitoring - everything operational
Production Ready: System must be serving traffic and fully functional at script completion
No Post-Deployment Tasks: No additional configuration, setup, or manual steps required
Complete Validation: Script validates entire system is operational before declaring success
Zero Mistakes Tolerance: Any error triggers automatic rollback to previous state

CRITICAL: Complete Self-Sufficiency - Auto-Install Everything:

Dependency Detection: Automatically detect all missing tools, packages, and dependencies
Auto-Installation: Install all missing prerequisites without user intervention
Platform Intelligence: Detect OS and use appropriate package managers (apt, yum, brew, apk, etc.)
Version Validation: Ensure installed versions meet minimum requirements for deployment
Tool Chain Setup: Install complete toolchain (Docker, Git, curl, wget, systemctl, etc.)
Runtime Installation: Install all required runtimes (Node.js, Python, Java, etc.) with correct versions
Database Installation: Install and configure required databases (MySQL, PostgreSQL, Redis, etc.)
Web Server Installation: Install and configure web servers (nginx, Apache) with optimal settings
Monitoring Tools: Install monitoring stack (Prometheus, Grafana, or alternatives)
Security Tools: Install security tools (fail2ban, ufw, SSL certificate management)
Network Tools: Install network utilities (iptables, netstat, ss, iperf) for optimization
Backup Tools: Install backup utilities and configure automated backup systems

CRITICAL: Hardware and Network Resource Detection and Optimization:

Hardware Detection: Detect CPU cores, RAM, disk space, and network interfaces using standard tools
Resource Optimization: Configure Docker, databases, and services based on detected hardware
Docker Configuration: Set Docker daemon limits, container resources based on available CPU/RAM
Network Optimization: Configure connection pools, timeouts based on detected network capacity
Storage Optimization: Configure disk I/O, cache sizes based on available storage type and space
Memory Allocation: Set JVM heaps, database buffers, cache sizes based on available RAM
CPU Utilization: Configure worker processes, thread pools based on available CPU cores
Pre-flight Validation: Test hardware meets minimum requirements before deployment begins

MANDATORY FIRST STEP - Investigation and Consolidation:

ALWAYS search for existing deployment scripts across entire codebase before creating new ones
Use comprehensive search: find . -name "*deploy*" -o -name "*build*" -o -name "*install*" -o -name "*.sh"
Investigate all found scripts: purpose, functionality, dependencies, and current usage
Analyze Git history to understand why each script was created and its evolution
Test each existing script in isolated environment to understand full functionality
Map all unique capabilities and consolidate into single ./deploy.sh
Preserve all working functionality - never lose capabilities during consolidation
Document what was consolidated from each script and why
Test consolidated script thoroughly in all environments before removing originals
Archive original scripts with restoration procedures before deletion
Update all references, documentation, and CI/CD pipelines to use new unified script
Validate that team members can execute same workflows with consolidated script
Only remove original scripts after 100% validation that consolidated version works

✅ Required Practices:
Complete Dependency Management and Auto-Installation:

Operating System Detection: Detect OS type and version using uname, /etc/os-release, lsb_release
Package Manager Detection: Identify available package managers (apt, yum, dnf, zypper, brew, pkg)
Dependency Scanning: Scan for all required tools and packages before starting deployment
Auto-Installation Logic: Install missing dependencies using appropriate package manager
Version Checking: Verify installed versions meet minimum requirements, upgrade if necessary
Alternative Installation: Use alternative installation methods if package manager fails (wget, curl, compile)
Tool Validation: Test that installed tools work correctly before proceeding
Path Configuration: Ensure all installed tools are in PATH and accessible
Permission Setup: Configure appropriate permissions for installed tools and services
Service Configuration: Configure installed services for automatic startup and optimal operation

Essential Tool Installation:
bash# Core system tools
install_core_tools() {
    local tools=("curl" "wget" "git" "unzip" "tar" "gzip" "systemctl" "ps" "netstat" "ss")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            install_package "$tool"
        fi
    done
}

# Docker installation
install_docker() {
    if ! command -v docker &> /dev/null; then
        case "$OS" in
            "ubuntu"|"debian")
                curl -fsSL https://get.docker.com | sh
                ;;
            "centos"|"rhel"|"fedora")
                curl -fsSL https://get.docker.com | sh
                ;;
            "alpine")
                apk add docker docker-compose
                ;;
        esac
        systemctl enable docker
        systemctl start docker
    fi
    validate_docker_installation
}

# Database installation
install_databases() {
    if [[ "$REQUIRES_MYSQL" == "true" ]] && ! command -v mysql &> /dev/null; then
        install_mysql_server
        configure_mysql_optimal_settings
    fi
    
    if [[ "$REQUIRES_POSTGRESQL" == "true" ]] && ! command -v psql &> /dev/null; then
        install_postgresql_server
        configure_postgresql_optimal_settings
    fi
    
    if [[ "$REQUIRES_REDIS" == "true" ]] && ! command -v redis-cli &> /dev/null; then
        install_redis_server
        configure_redis_optimal_settings
    fi
}

# Runtime installation
install_runtimes() {
    if [[ "$REQUIRES_NODEJS" == "true" ]] && ! command -v node &> /dev/null; then
        install_nodejs_lts
        validate_nodejs_version
    fi
    
    if [[ "$REQUIRES_PYTHON" == "true" ]] && ! command -v python3 &> /dev/null; then
        install_python3_and_pip
        validate_python_version
    fi
    
    if [[ "$REQUIRES_JAVA" == "true" ]] && ! command -v java &> /dev/null; then
        install_openjdk
        validate_java_version
    fi
}
Platform-Specific Installation Logic:

Ubuntu/Debian: Use apt update && apt install -y for package installation
CentOS/RHEL/Fedora: Use yum install -y or dnf install -y for package installation
Alpine Linux: Use apk add for package installation
macOS: Use brew install if Homebrew available, otherwise use manual installation
Amazon Linux: Use yum install -y with Amazon Linux-specific repositories
SUSE/openSUSE: Use zypper install -y for package installation
Arch Linux: Use pacman -S for package installation
FreeBSD: Use pkg install for package installation
Generic Unix: Use manual installation methods (wget, curl, compile from source)

Hardware Resource Detection and Optimization:

CPU Detection: Use nproc, /proc/cpuinfo to detect cores, architecture, features
Memory Detection: Use free, /proc/meminfo to detect total/available RAM
Storage Detection: Use df, lsblk to detect disk space, filesystem types, mount points
Network Detection: Use ip, ethtool to detect network interfaces and capabilities
Performance Testing: Run basic performance tests (dd for disk, iperf for network) to establish baselines
Resource Calculation: Calculate optimal settings based on detected resources (e.g., DB buffer = 25% RAM)
Docker Daemon Config: Set --default-ulimit, --storage-driver based on system capabilities
Container Limits: Set --memory, --cpus for each container based on available resources
Database Tuning: Configure MySQL/PostgreSQL buffer pools, connection limits based on RAM
Web Server Config: Set worker processes, connection limits based on CPU cores

Docker and Container Optimization:

Docker Installation: Auto-install Docker if not present using official installation scripts
Docker Compose Installation: Install Docker Compose if not available
Storage Driver Selection: Choose optimal storage driver (overlay2, devicemapper) based on filesystem
Memory Limits: Set container memory limits to prevent OOM kills while maximizing usage
CPU Limits: Set CPU limits to ensure fair resource sharing without waste
Container Placement: Use Docker Compose resource constraints for optimal container placement
Health Checks: Configure comprehensive health checks for all containers
Restart Policies: Set appropriate restart policies based on service criticality
Network Configuration: Configure Docker networks for optimal performance and security
Volume Configuration: Configure volumes with appropriate permissions and performance settings
Registry Configuration: Configure container registry with optimal caching and authentication

Network Resource Optimization:

Network Tools Installation: Install iperf3, netstat, ss, tcpdump for network analysis
Bandwidth Detection: Test available bandwidth using simple tools (wget, curl with large files)
Latency Testing: Test network latency to external services and databases
Connection Pool Sizing: Configure database connection pools based on CPU cores and expected load
Timeout Configuration: Set timeouts based on measured network latency + buffer
Load Balancer Config: Configure nginx/HAProxy with optimal worker processes and connections
DNS Configuration: Configure DNS caching and resolution for optimal performance
Firewall Configuration: Configure iptables/ufw rules with performance impact
TCP Tuning: Tune TCP buffer sizes and window scaling for optimal throughput
SSL Configuration: Configure SSL with optimal cipher suites and session management
CDN Configuration: Configure CDN settings based on geographic deployment location

Pre-flight Checks and Validation:

Tool Availability: Verify all required tools are installed and functional
Version Compatibility: Check all tool versions meet minimum requirements
Permission Validation: Verify script has necessary permissions for all operations
Minimum Requirements: Verify minimum CPU, RAM, disk space, network connectivity
Port Availability: Check all required ports are available before starting services
External Connectivity: Test connectivity to external services, databases, APIs
Storage Performance: Test disk I/O performance meets minimum requirements
Memory Availability: Ensure sufficient memory available for all planned services
Network Connectivity: Validate network connectivity and DNS resolution
Security Requirements: Verify system meets basic security requirements
Backup Validation: Verify backup storage is accessible and has sufficient space

Complete End-to-End Automation:

System Preparation: Install all required packages, create users, configure system services
Database Setup: Install, configure, and initialize databases with schemas and initial data
Application Deployment: Build, deploy, and configure all application services
Web Server Setup: Install and configure nginx/Apache with SSL and optimal settings
Security Configuration: Configure firewalls, SSL certificates, basic security hardening
Monitoring Setup: Deploy basic monitoring (e.g., Prometheus, Grafana) with essential metrics
Backup Configuration: Setup automated backups with tested restore procedures
Service Integration: Configure and test all service-to-service communication
Health Validation: Validate all services are running and responding correctly
Performance Testing: Run basic performance tests to ensure system meets requirements
Documentation Generation: Generate basic operational documentation and access information
Team Notification: Send notification with deployment status and access information

Investigation and Consolidation (MANDATORY FIRST STEP):

Search for existing scripts: deployment, build, install, setup, CI/CD scripts
Fresh Server Scenario: If no existing scripts found, proceed directly with deployment
Existing Scripts Scenario: If scripts found, require consolidation before proceeding
Analyze each script's functionality using standard tools (grep, awk, bash analysis)
Map dependencies and understand integration points with existing systems
Test existing scripts in isolated environments to understand behavior
Consolidate functionality by merging working code into single script
Preserve all working functionality during consolidation process
Document consolidation decisions and archive original scripts safely
Test consolidated script thoroughly before removing originals
Update all references and documentation to point to consolidated script

Zero Assumptions Architecture:

Package Management: Auto-detect and use appropriate package manager (apt, yum, brew)
Dependency Installation: Install all required system dependencies automatically
User Management: Create required system users with appropriate permissions
Directory Structure: Create all required directories with correct permissions
Service Configuration: Configure all system services for automatic startup
Environment Setup: Configure environment variables and system paths
Security Setup: Configure basic security (firewall, user permissions, SSL)
Database Initialization: Initialize databases with required schemas and users
Application Configuration: Generate and deploy all application configuration files
Network Setup: Configure network interfaces, routing, and DNS as needed

Environment Management:

Environment Detection: Detect environment based on hostname, network, or explicit flags
Configuration Loading: Load environment-specific configurations from standard locations
Resource Allocation: Adjust resource allocation based on environment (dev uses less resources)
Security Policies: Apply appropriate security policies for each environment
Monitoring Configuration: Configure monitoring appropriate for environment sensitivity
Backup Policies: Apply backup and retention policies appropriate for environment
Performance Settings: Apply performance tuning appropriate for environment load
Integration Configuration: Configure external service integrations per environment
Feature Flags: Configure feature flags and toggles per environment
Scaling Policies: Configure auto-scaling appropriate for environment requirements

Self-Update Mechanism:

Version Checking: Check for script updates against source repository
Backup Current: Backup current script before any updates
Download and Validate: Download new version and validate syntax
Integrity Verification: Verify download integrity using checksums
Update and Restart: Update script and restart with new version
Configuration Updates: Update configuration files and templates
Dependency Updates: Check and update system dependencies
Container Updates: Pull new container images and update configurations
Rollback Capability: Rollback to previous version if update fails
Change Documentation: Document what changed and why

Comprehensive Error Handling:

Structured Exit Codes: Use specific exit codes for different failure types
Error Logging: Log all errors with timestamps and context information
Retry Logic: Implement retry logic for transient failures (network, etc.)
Graceful Degradation: Continue with non-critical failures, abort on critical ones
Rollback Triggers: Automatically trigger rollback on specific error conditions
Error Notification: Send notifications for errors requiring human attention
Recovery Procedures: Document and automate recovery procedures for common failures
Timeout Handling: Handle timeouts for all external operations
Resource Cleanup: Clean up resources on failure to prevent resource leaks
State Validation: Validate system state before and after critical operations

Rollback and Recovery:

State Snapshots: Create snapshots of critical system state before changes
Database Backups: Backup databases before schema changes or data operations
Configuration Backups: Backup all configuration files before modifications
Service State Tracking: Track which services were started/modified for rollback
Atomic Operations: Use atomic operations where possible (database transactions, etc.)
Rollback Scripts: Generate rollback scripts during deployment for easy recovery
Validation During Rollback: Validate system state during rollback process
Cleanup Procedures: Clean up any resources created during failed deployment
Service Restoration: Restore services to previous running state
Data Integrity: Ensure data integrity during rollback operations

Validation and Reporting:

Pre-deployment Checks: Validate system requirements and dependencies
Deployment Monitoring: Monitor deployment progress with regular status updates
Service Health Checks: Validate all services are healthy and responding
Performance Validation: Basic performance testing to ensure acceptable response times
Security Validation: Basic security checks (open ports, permissions, SSL)
Integration Testing: Test critical integrations with external services
End-to-End Testing: Run basic end-to-end tests of critical user workflows
Deployment Report: Generate comprehensive deployment report with metrics
Access Information: Provide access URLs, credentials, and operational information
Next Steps: Document any recommended next steps or optimizations

🚫 Forbidden Practices:
Dependency Management Violations:

Assuming any tools or packages are pre-installed on target systems
Requiring manual installation of dependencies before script execution
Using tools without checking if they're available and installing if missing
Failing to validate that installed tools work correctly before using them
Using hardcoded paths to tools without checking if they exist
Ignoring different package managers across different operating systems
Installing packages without verifying successful installation
Using deprecated or insecure installation methods
Installing packages without configuring them appropriately
Failing to handle installation failures gracefully with alternatives

Realistic Constraint Violations:

Using theoretical or experimental technologies not available in standard distributions
Requiring specialized hardware or software not commonly available
Implementing features that require months of development work
Using AI/ML capabilities that don't exist in standard deployment tools
Requiring manual installation of complex custom tools before script execution
Using cloud-specific features that don't work on-premises or other clouds
Implementing security features that require specialized security hardware
Using performance optimization techniques that require kernel modifications
Requiring enterprise licenses or paid tools for basic functionality
Implementing features that require specialized networking equipment

Hardware and Resource Optimization Violations:

Using hardcoded resource limits that ignore available hardware
Deploying Docker without configuring resource limits and usage
Ignoring available CPU cores when configuring worker processes
Using fixed memory allocations regardless of available RAM
Configuring databases without considering available memory
Using default network settings without testing available bandwidth
Ignoring disk performance characteristics when configuring storage
Setting timeouts without considering actual network latency
Using single-threaded processes on multi-core systems
Configuring services without considering system resource capacity

Complete Automation Violations:

Requiring manual input, confirmation, or intervention during deployment
Deploying partial systems that require additional manual setup
Using deployment procedures that pause and wait for human action
Creating deployments that require manual post-deployment configuration
Using scripts that fail silently and require manual troubleshooting
Deploying without validating that all services are actually working
Creating systems that are deployed but not ready to serve traffic
Using deployment procedures that require specialized knowledge to execute
Deploying without comprehensive error handling and automatic recovery
Creating deployments that leave systems in inconsistent or broken states

Investigation and Consolidation Violations:

Creating new deployment scripts without searching for existing ones
Ignoring existing deployment automation when building new solutions
Consolidating scripts without understanding their complete functionality
Removing existing scripts without thorough testing of replacements
Creating consolidated scripts that lose important existing capabilities
Skipping analysis of why existing scripts were created originally
Consolidating without documenting what functionality came from where
Removing scripts without proper backup and restoration procedures
Creating solutions that break existing team workflows and procedures
Ignoring integration with existing CI/CD and operational tools

Validation Criteria:
Dependency Management and Auto-Installation Validation:

All required tools and packages automatically detected and installed
Installation works correctly across different operating systems and package managers
Version validation ensures all tools meet minimum requirements
Alternative installation methods work when package managers fail
Tool functionality validated after installation before proceeding with deployment
PATH and environment properly configured for all installed tools
Installation failures handled gracefully with appropriate error messages
No manual intervention required for any dependency installation
Installation process documented in deployment logs for troubleshooting
Rollback procedures can cleanly remove installed dependencies if needed

Hardware and Resource Optimization Validation:

System hardware completely detected and documented (CPU, RAM, disk, network)
Docker configured with appropriate resource limits based on available hardware
Database configurations optimized for available memory and CPU
Web server configurations optimized for available CPU cores
Network configurations optimized for detected bandwidth and latency
Container resource limits prevent resource contention while maximizing utilization
Performance testing validates optimization settings improve performance
Resource monitoring confirms optimal utilization without resource exhaustion
System performs better than default configurations under realistic load
Resource scaling works correctly as system load increases

Complete End-to-End Validation:

Single command execution results in fully functional system serving requests
Database is operational with all required schemas, users, and initial data
Application services are running and responding to requests correctly
Web server is serving content with proper SSL configuration
All required system services are running and configured for auto-start
Monitoring is operational and collecting metrics from all services
Backup systems are configured and initial backups have been created
Security configurations are applied and basic security tests pass
Integration with external services is working correctly
System can handle expected production traffic loads
No manual intervention was required during any phase of deployment
System is ready for production use immediately after script completion

Investigation and Consolidation Validation:

Comprehensive search completed for all existing deployment-related scripts
Fresh Server: No existing scripts found, deployment proceeds normally
Existing Scripts: All scripts analyzed and consolidated successfully
All functionality from original scripts preserved in consolidated version
Consolidated script tested thoroughly in all relevant environments
Team workflows continue to work with consolidated script
All references and documentation updated to use consolidated script
Original scripts properly archived with tested restoration procedures
No regression in deployment capabilities or functionality
Team training completed on consolidated script usage

Functional Validation:

Script executes successfully on fresh systems without any manual setup
All environment configurations deploy correctly with appropriate settings
Self-update mechanism works reliably with proper backup and validation
Error handling provides clear information and triggers appropriate recovery
Rollback procedures successfully restore system to previous working state
All validation checks accurately detect problems before they cause issues
Deployment reporting provides comprehensive status and access information
Script integrates properly with existing CI/CD and monitoring systems
Performance meets established requirements under realistic load conditions
Security configurations meet basic security requirements for environment

Operational Validation:

Any team member can execute deployment without specialized training
Deployment logs provide sufficient detail for troubleshooting any issues
Monitoring provides real-time visibility into deployment progress and system health
Documentation is complete and enables effective operational management
Emergency procedures work correctly for common failure scenarios
System maintenance procedures are documented and tested
Team has been trained on operational procedures and troubleshooting
Integration with existing operational tools and procedures is functional
Compliance requirements are met for the target environment
Long-term maintenance and support procedures are established and documented


*Last Updated: 2025-08-30 00:00:00 UTC - For the infrastructure based in /opt/sutazaiapp/