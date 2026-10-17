import * as ort from "onnxruntime-web/webgpu";
import type { BBox } from "./processOutput";

const EYE_SIZE = 224; // imgsz del clasificador de ojo

// Proporciones respecto a la bbox de la cabeza (0 a 1). AJÚSTALAS mirando el recorte.
const EYE_CENTER_X = 0.65; // 0.65 = ojo izquierdo de la persona (derecha en imagen)
const EYE_CENTER_Y = 0.40; // altura del ojo desde la parte superior de la caja
const EYE_SIDE = 0.35;     // lado del recorte cuadrado, como fracción del ancho de la caja

// Canvas y buffer creados una sola vez
const canvas = document.createElement("canvas");
canvas.width = EYE_SIZE;
canvas.height = EYE_SIZE;
const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
const floatData = new Float32Array(3 * EYE_SIZE * EYE_SIZE);
const plane = EYE_SIZE * EYE_SIZE;

export function eyeCropToTensor(video: HTMLVideoElement, box: BBox): ort.Tensor {
    const vw = video.videoWidth;
    const vh = video.videoHeight;

    // Cuadrado centrado en la posición estimada del ojo
    const side = Math.min(box.w * EYE_SIDE, vw, vh);
    const cx = box.x + box.w * EYE_CENTER_X;
    const cy = box.y + box.h * EYE_CENTER_Y;

    // Si se sale del vídeo, se desplaza dentro en vez de recortarse (mantiene el cuadrado)
    const sx = Math.min(Math.max(cx - side / 2, 0), vw - side);
    const sy = Math.min(Math.max(cy - side / 2, 0), vh - side);

    ctx.drawImage(video, sx, sy, side, side, 0, 0, EYE_SIZE, EYE_SIZE);

    // RGBA intercalado -> NCHW planar, normalizado a [0, 1]
    const { data } = ctx.getImageData(0, 0, EYE_SIZE, EYE_SIZE);
    for (let i = 0; i < plane; i++) {
        floatData[i]             = data[i * 4]     / 255;
        floatData[i + plane]     = data[i * 4 + 1] / 255;
        floatData[i + 2 * plane] = data[i * 4 + 2] / 255;
    }

    return new ort.Tensor("float32", floatData, [1, 3, EYE_SIZE, EYE_SIZE]);
}

// Para debug
export const eyeDebugCanvas = canvas;