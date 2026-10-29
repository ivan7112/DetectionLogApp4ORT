import onnx

# Cargar el modelo ONNX
model = onnx.load("./public/head_detector.onnx")

# Buscar las clases en los metadatos
for meta in model.metadata_props:
    if meta.key == "names":
        # Devuelve un string en formato JSON/Diccionario, ej: {0: 'person', 1: 'bicycle', ...}
        print("Clases encontradas:", meta.value)