LEAF_CLASSES = [
    "Healthy",
    "Mosaic",
    "Rust",
    "Septoria_brown_spot",
    "Frogeye_leaf_spot",
    "Caterpillar_Semi_looper",
]

# UAV dataset only has 4 folders per the document (Healthy + 2 diseases + pest) —
# it does NOT include Frogeye/Septoria as separate UAV classes. We use the
# leaf taxonomy as the single output space (finer-grained), and the UAV
# branch contributes field-level context even when its own 4-class folder
# structure doesn't cover all 6 leaf classes.
UAV_CLASSES = ["Healthy", "Mosaic", "Rust", "Caterpillar_Semi_looper"]

IMAGE_SIZE = 224
