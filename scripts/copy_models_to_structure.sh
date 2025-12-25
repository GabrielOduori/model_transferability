#!/bin/bash
# Copy existing trained models to the new models/ directory structure

echo "🔄 Copying trained models to models/ directory..."
echo ""

# Source paths (adjust these to your actual model locations)
FUSIONGP_SOURCE="gam_ssm_lur/fusionGP2/fusiongp/notebooks/fusiongp_model.pt"
GAM_SSM_LUR_SOURCE=""  # TODO: Add path when found

# Destination paths
FUSIONGP_DEST="models/fusiongp/dublin/fusiongp_dublin_trained.pt"
GAM_SSM_LUR_DEST="models/gam_ssm_lur/dublin/gam_ssm_lur_dublin_trained.pt"

# Copy FusionGP
if [ -f "$FUSIONGP_SOURCE" ]; then
    echo "📦 Copying FusionGP model..."
    cp "$FUSIONGP_SOURCE" "$FUSIONGP_DEST"
    echo "   ✓ Copied to: $FUSIONGP_DEST"

    # Show file info
    ls -lh "$FUSIONGP_DEST"
else
    echo "⚠️  FusionGP model not found at: $FUSIONGP_SOURCE"
fi

echo ""

# Copy GAM-SSM-LUR
if [ -f "$GAM_SSM_LUR_SOURCE" ]; then
    echo "📦 Copying GAM-SSM-LUR model..."
    cp "$GAM_SSM_LUR_SOURCE" "$GAM_SSM_LUR_DEST"
    echo "   ✓ Copied to: $GAM_SSM_LUR_DEST"

    # Show file info
    ls -lh "$GAM_SSM_LUR_DEST"
else
    echo "⚠️  GAM-SSM-LUR model path not set or file not found"
    echo "   Please update GAM_SSM_LUR_SOURCE in this script"
fi

echo ""
echo "✅ Model copying complete!"
echo ""
echo "Next steps:"
echo "1. Extract training data and save as .pkl files"
echo "2. Create metadata.json files for each model"
echo "3. Update run_real_model_transfer.py to use models/ paths"
