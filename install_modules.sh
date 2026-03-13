#!/bin/bash

echo "🚀 Building Chamfer3D..."
cd extensions/chamfer_dist
python setup.py install
cd ../..

echo "🚀 Building Expansion Penalty..."
cd extensions/expansion_penalty
python setup.py install
cd ../..

echo "🚀 Building PointNet2 Ops..."
cd pointnet2_ops_lib
python setup.py install
cd ..

echo "✅ All custom CUDA modules compiled successfully!"