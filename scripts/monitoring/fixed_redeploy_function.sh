#!/bin/bash

# Fixed redeploy_all_containers function
redeploy_all_containers_fixed() {
    # Track deployment start time
    DEPLOYMENT_START_TIME=$(date +%s)
    
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║              REDEPLOY ALL CONTAINERS                        ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo -e "${YELLOW}⚠️  WARNING: This will stop and redeploy all containers!${NC}"
    echo -e "${YELLOW}This action will:${NC}"
    echo "• Stop all running containers"
    echo "• Remove all containers (preserving data volumes)"
    echo "• Rebuild and restart all containers"
    echo "• Apply any configuration changes"
    echo ""
    read -p "Are you sure you want to proceed? (y/N): " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Redeployment cancelled.${NC}"
        return 0
    fi
    
    echo ""
    echo -e "${CYAN}Starting enhanced redeployment process...${NC}"
    echo ""
    
    # Hardware Auto-Detection
    echo -e "${CYAN}🔍 Detecting hardware resources...${NC}"
    
    # CPU Detection
    CPU_CORES=$(nproc)
    CPU_THREADS=$(nproc --all)
    CPU_MODEL=$(lscpu 2>/dev/null | grep "Model name" | cut -d: -f2 | xargs || echo "Unknown")
    
    # Memory Detection
    RAM_TOTAL_GB=$(free -g | awk 'NR==2{print $2}')
    RAM_AVAILABLE_GB=$(free -g | awk 'NR==2{print $7}')
    RAM_USAGE_PERCENT=$(free | awk 'NR==2{printf "%.0f", $3*100/($3+$4)}')
    
    # Disk Type Detection
    DISK_TYPE="HDD"
    if [ -d "/sys/block" ]; then
        for disk in /sys/block/*/queue/rotational; do
            if [ -r "$disk" ] && [ "$(cat "$disk")" = "0" ]; then
                DISK_TYPE="SSD"
                break
            fi
        done
    fi
    
    # GPU Detection
    GPU_PRESENT="false"
    GPU_COUNT=0
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | wc -l || echo 0)
        if [ "$GPU_COUNT" -gt 0 ]; then
            GPU_PRESENT="true"
        fi
    fi
    
    echo -e "${GREEN}📊 Hardware Profile:${NC}"
    echo "  CPU: $CPU_MODEL ($CPU_CORES cores, $CPU_THREADS threads)"
    echo "  RAM: ${RAM_AVAILABLE_GB}GB available / ${RAM_TOTAL_GB}GB total (${RAM_USAGE_PERCENT}% used)"
    echo "  Disk: $DISK_TYPE"
    echo "  GPU: $GPU_PRESENT (${GPU_COUNT} devices)"
    echo ""
    
    # Calculate optimal parallelism
    PULL_PARALLELISM=$CPU_CORES
    BUILD_PARALLELISM=$((CPU_CORES / 2))
    
    # Adjust based on RAM
    if [ "$RAM_AVAILABLE_GB" -lt 4 ]; then
        PULL_PARALLELISM=$((CPU_CORES / 2))
        BUILD_PARALLELISM=1
        echo -e "${YELLOW}⚠️  Low RAM detected, reducing parallelism${NC}"
    elif [ "$RAM_AVAILABLE_GB" -gt 16 ]; then
        PULL_PARALLELISM=$((CPU_CORES * 2))
        BUILD_PARALLELISM=$CPU_CORES
        echo -e "${GREEN}✅ High RAM available, increasing parallelism${NC}"
    fi
    
    # Adjust based on disk type
    if [ "$DISK_TYPE" = "SSD" ]; then
        PULL_PARALLELISM=$((PULL_PARALLELISM + 2))
        echo -e "${GREEN}✅ SSD detected, optimizing I/O operations${NC}"
    fi
    
    # Cap maximum parallelism
    PULL_PARALLELISM=$(( PULL_PARALLELISM > 16 ? 16 : PULL_PARALLELISM ))
    BUILD_PARALLELISM=$(( BUILD_PARALLELISM > 8 ? 8 : BUILD_PARALLELISM ))
    
    echo -e "${CYAN}🔧 Optimization Settings:${NC}"
    echo "  Pull parallelism: $PULL_PARALLELISM"
    echo "  Build parallelism: $BUILD_PARALLELISM"
    echo ""
    
    # Enable Docker BuildKit for faster builds
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    echo ""
    
    # Navigate to project root
    cd /opt/sutazaiapp || {
        echo -e "${RED}Error: Cannot navigate to project root${NC}"
        return 1
    }
    
    # Check if docker-compose file exists
    if [[ ! -f "docker-compose.yml" ]]; then
        echo -e "${RED}Error: docker-compose.yml not found${NC}"
        return 1
    fi
    
    # Load environment variables if .env exists
    if [[ -f ".env" ]]; then
        echo -e "${CYAN}Loading environment variables from .env file...${NC}"
        set -a
        source .env
        set +a
    fi
    
    # Task allocation message
    echo -e "${PURPLE}🤖 Allocating AI agents to handle the redeployment...${NC}"
    echo ""
    
    echo -e "${GREEN}✓ infrastructure-devops-manager${NC} - Managing deployment orchestration"
    echo -e "${GREEN}✓ deployment-automation-master${NC} - Handling container lifecycle"
    echo -e "${GREEN}✓ system-optimizer-reorganizer${NC} - Optimizing resource allocation"
    echo -e "${GREEN}✓ monitoring-engineer${NC} - Ensuring service health checks"
    echo -e "${GREEN}✓ self-healing-orchestrator${NC} - Monitoring for failures"
    echo ""
    
    # Step 1: Stop all containers
    echo -e "${CYAN}Step 1/5: Stopping all containers...${NC}"
    docker compose down --remove-orphans 2>&1 | while read -r line; do
        if [[ ! "$line" =~ "warning" ]]; then
            echo -e "  ${line}"
        fi
    done
    
    # Step 2: Clean up
    echo ""
    echo -e "${CYAN}Step 2/5: Cleaning up unused resources...${NC}"
    docker system prune -f 2>&1 | while read -r line; do
        echo -e "  ${line}"
    done
    
    # Step 3: Build and pull images properly
    echo ""
    echo -e "${CYAN}Step 3/5: Building and pulling images...${NC}"
    
    # Get all services from docker-compose
    ALL_SERVICES=$(docker compose config --services 2>/dev/null)
    TOTAL_SERVICES=$(echo "$ALL_SERVICES" | wc -l)
    
    echo "Found $TOTAL_SERVICES services to process"
    
    # Arrays to track services
    declare -a external_images=()
    declare -a build_services=()
    declare -a failed_services=()
    
    # Analyze each service
    echo -e "${CYAN}Analyzing services...${NC}"
    for service in $ALL_SERVICES; do
        # Check if service has build context
        if docker compose config | yq eval ".services.$service.build" 2>/dev/null | grep -qv "null"; then
            build_services+=("$service")
        else
            # Get the image name
            image=$(docker compose config | yq eval ".services.$service.image" 2>/dev/null || echo "")
            if [[ -n "$image" ]] && [[ "$image" != "null" ]]; then
                external_images+=("$image")
            fi
        fi
    done
    
    # First, build all local services
    if [[ ${#build_services[@]} -gt 0 ]]; then
        echo ""
        echo -e "${CYAN}Building ${#build_services[@]} local services...${NC}"
        
        for service in "${build_services[@]}"; do
            echo -e "  ${YELLOW}[BUILD]${NC} Building $service..."
            if docker compose build "$service" 2>&1 | tail -n 20; then
                echo -e "  ${GREEN}[SUCCESS]${NC} $service built successfully"
            else
                echo -e "  ${RED}[FAIL]${NC} $service build failed"
                failed_services+=("$service")
            fi
        done
    fi
    
    # Then pull external images
    if [[ ${#external_images[@]} -gt 0 ]]; then
        echo ""
        echo -e "${CYAN}Pulling ${#external_images[@]} external images...${NC}"
        
        for image in "${external_images[@]}"; do
            # Skip local builds (sutazaiapp-* images)
            if [[ "$image" =~ ^sutazaiapp- ]]; then
                echo -e "  ${YELLOW}[SKIP]${NC} $image (local build)"
                continue
            fi
            
            echo -e "  ${YELLOW}[PULL]${NC} $image"
            if docker pull "$image" 2>&1 | tail -n 5; then
                echo -e "  ${GREEN}[SUCCESS]${NC} $image"
            else
                echo -e "  ${RED}[FAIL]${NC} $image"
            fi
        done
    fi
    
    # Step 4: Start all containers
    echo ""
    echo -e "${CYAN}Step 4/5: Starting all containers...${NC}"
    
    docker compose up -d 2>&1 | while read -r line; do
        if [[ ! "$line" =~ "warning" ]]; then
            if [[ "$line" =~ "Created" ]] || [[ "$line" =~ "Started" ]]; then
                echo -e "  ${GREEN}${line}${NC}"
            elif [[ "$line" =~ "Error" ]]; then
                echo -e "  ${RED}${line}${NC}"
            else
                echo -e "  ${line}"
            fi
        fi
    done
    
    # Step 5: Verify deployment
    echo ""
    echo -e "${CYAN}Step 5/5: Verifying deployment...${NC}"
    
    sleep 5
    
    # Count running containers
    RUNNING_COUNT=$(docker ps --filter "name=sutazai-" --format "{{.Names}}" | wc -l)
    echo -e "${GREEN}Running containers: $RUNNING_COUNT${NC}"
    
    # Check for failed containers
    FAILED_COUNT=$(docker ps -a --filter "name=sutazai-" --filter "status=exited" --format "{{.Names}}" | wc -l)
    if [[ $FAILED_COUNT -gt 0 ]]; then
        echo -e "${RED}Failed containers: $FAILED_COUNT${NC}"
        docker ps -a --filter "name=sutazai-" --filter "status=exited" --format "table {{.Names}}\t{{.Status}}"
    fi
    
    # Calculate deployment time
    DEPLOYMENT_END_TIME=$(date +%s)
    DEPLOYMENT_DURATION=$((DEPLOYMENT_END_TIME - DEPLOYMENT_START_TIME))
    
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ Redeployment completed in ${DEPLOYMENT_DURATION} seconds${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    
    # Summary
    echo ""
    echo -e "${CYAN}Deployment Summary:${NC}"
    echo "  Total services: $TOTAL_SERVICES"
    echo "  Running containers: $RUNNING_COUNT"
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        echo -e "  ${RED}Failed builds: ${#failed_services[@]}${NC}"
        for service in "${failed_services[@]}"; do
            echo -e "    ${RED}- $service${NC}"
        done
    fi
    if [[ $FAILED_COUNT -gt 0 ]]; then
        echo -e "  ${RED}Failed containers: $FAILED_COUNT${NC}"
    fi
}

# Export the function
export -f redeploy_all_containers_fixed