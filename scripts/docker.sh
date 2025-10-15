#!/bin/bash
# Docker helper scripts for QVP Platform

# Build the Docker image
build() {
    echo "Building QVP Docker image..."
    docker build -t qvp-platform:latest .
}

# Run the demo in Docker
run-demo() {
    echo "Running QVP demo in Docker..."
    docker run --rm -v "$(pwd)/data:/app/data" qvp-platform:latest
}

# Start all services with docker-compose
start-all() {
    echo "Starting all QVP services..."
    docker-compose up -d
}

# Stop all services
stop-all() {
    echo "Stopping all QVP services..."
    docker-compose down
}

# View logs
logs() {
    service=${1:-qvp-app}
    docker-compose logs -f "$service"
}

# Execute command in container
exec-bash() {
    service=${1:-qvp-app}
    docker-compose exec "$service" bash
}

# Rebuild and restart
rebuild() {
    echo "Rebuilding and restarting services..."
    docker-compose down
    docker-compose build --no-cache
    docker-compose up -d
}

# Show status
status() {
    docker-compose ps
}

# Main command dispatcher
case "$1" in
    build)
        build
        ;;
    run-demo)
        run-demo
        ;;
    start)
        start-all
        ;;
    stop)
        stop-all
        ;;
    logs)
        logs "$2"
        ;;
    exec)
        exec-bash "$2"
        ;;
    rebuild)
        rebuild
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {build|run-demo|start|stop|logs|exec|rebuild|status}"
        echo ""
        echo "Commands:"
        echo "  build       - Build the Docker image"
        echo "  run-demo    - Run demo script in Docker"
        echo "  start       - Start all services with docker-compose"
        echo "  stop        - Stop all services"
        echo "  logs [svc]  - View logs (default: qvp-app)"
        echo "  exec [svc]  - Execute bash in container (default: qvp-app)"
        echo "  rebuild     - Rebuild and restart all services"
        echo "  status      - Show service status"
        exit 1
        ;;
esac
