#!/bin/bash

# Run Workfront Demo on Cluster
# This creates a web interface accessible from your browser

echo "🚀 Starting Workfront AI Demo"
echo "============================="

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit
fi

# Get cluster IP for external access
CLUSTER_IP=$(hostname -I | awk '{print $1}')
PORT=8501

echo "🌐 Starting Streamlit server..."
echo "📍 Access URLs:"
echo "   Internal: http://localhost:$PORT"
echo "   External: http://$CLUSTER_IP:$PORT"
echo ""
echo "🔧 To access from your laptop:"
echo "   1. Open browser and go to: http://$CLUSTER_IP:$PORT"
echo "   2. Or use SSH tunnel: ssh -L $PORT:localhost:$PORT [your-cluster]"
echo ""
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Run Streamlit with external access
streamlit run workfront_demo.py \
    --server.address 0.0.0.0 \
    --server.port $PORT \
    --server.headless true \
    --browser.gatherUsageStats false 