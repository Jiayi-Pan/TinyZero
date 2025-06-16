#!/bin/bash

echo "🚀 Starting Workfront AI Assistant Terminal Demo"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "verl/utils/dataset/workfront_data.json" ]; then
    echo "⚠️  Please run from the TinyZeroRL directory"
    exit 1
fi

# Check Python environment
echo "🔍 Checking Python environment..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || {
    echo "❌ PyTorch not found. Please install requirements."
    exit 1
}

echo "✅ Environment ready!"
echo ""

# Run the demo
python3 terminal_demo.py

echo ""
echo "�� Demo completed!" 
