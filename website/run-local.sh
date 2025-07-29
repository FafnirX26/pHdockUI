#!/bin/bash

echo "🚀 Starting pHdockUI Website Locally..."
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if running with Docker Compose
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

# Option 1: Run with Docker Compose (recommended)
echo "Option 1: Run with Docker Compose (recommended)"
echo "----------------------------------------------"
echo "Run: $COMPOSE_CMD up"
echo ""

# Option 2: Run without Docker
echo "Option 2: Run without Docker"
echo "----------------------------"
echo ""
echo "Terminal 1 - Backend:"
echo "  cd backend"
echo "  pip install -r requirements.txt"
echo "  python main.py"
echo ""
echo "Terminal 2 - Frontend:"
echo "  npm install"
echo "  npm run dev"
echo ""

# Option 3: Quick start (no backend)
echo "Option 3: Quick frontend-only demo"
echo "----------------------------------"
echo "  npm install"
echo "  npm run dev"
echo ""
echo "Note: The interactive interface won't work without the backend"
echo ""

# Ask user preference
read -p "Choose option (1/2/3): " choice

case $choice in
    1)
        echo "Starting with Docker Compose..."
        $COMPOSE_CMD up
        ;;
    2)
        echo "Please follow the manual steps above in separate terminals."
        ;;
    3)
        echo "Starting frontend only..."
        npm install
        npm run dev
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac 