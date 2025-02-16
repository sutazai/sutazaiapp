setup_postgres() {
    # Enhanced container management
    if ! docker ps -a | grep -q postgresql; then
        echo "Initializing new PostgreSQL container..."
        docker run --name postgresql \
            -e POSTGRES_USER=myuser \
            -e POSTGRES_PASSWORD=mypassword \
            -p 5432:5432 \
            -d postgres:latest
    else
        if ! docker ps | grep -q postgresql; then
            echo "Restarting existing PostgreSQL container..."
            docker start postgresql
        fi
    fi

    # Wait for database readiness
    echo "Waiting for PostgreSQL initialization..."
    until docker exec postgresql pg_isready -U postgres &> /dev/null; do
        sleep 2
    done

    # Existing database setup
    docker exec postgresql psql -U myuser -c "CREATE DATABASE ai_system;" || true
    docker exec postgresql psql -U myuser -d ai_system -c \
        "CREATE USER ai_user WITH ENCRYPTED PASSWORD 'secure_password';"
    docker exec postgresql psql -U myuser -d ai_system -c \
        "GRANT ALL PRIVILEGES ON DATABASE ai_system TO ai_user;"

    # Apply migrations
    echo "Running database migrations..."
    alembic upgrade head
}

install_docker() {
    # Install Docker based on OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case $ID in
            ubuntu|debian)
                sudo apt-get update
                sudo apt-get install -y \
                    apt-transport-https \
                    ca-certificates \
                    curl \
                    gnupg \
                    lsb-release
                curl -fsSL https://download.docker.com/linux/$ID/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
                echo \
                    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/$ID \
                    $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
                sudo apt-get update
                sudo apt-get install -y docker-ce docker-ce-cli containerd.io
                ;;
            centos|rhel)
                sudo yum install -y yum-utils
                sudo yum-config-manager \
                    --add-repo \
                    https://download.docker.com/linux/centos/docker-ce.repo
                sudo yum install -y docker-ce docker-ce-cli containerd.io
                ;;
            *)
                echo "Unsupported OS. Please install Docker manually."
                exit 1
                ;;
        esac
        
        # Start and enable Docker service
        sudo systemctl start docker
        sudo systemctl enable docker
        
        # Add current user to docker group
        sudo usermod -aG docker $USER
        newgrp docker
    else
        echo "Cannot detect OS. Please install Docker manually."
        exit 1
    fi
}